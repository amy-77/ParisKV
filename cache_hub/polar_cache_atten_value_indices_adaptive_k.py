
"""
Adaptive top-k/value-weighted:
    both value (K & indices) are adaptive.
   - **每一步都重算 adaptive_k**

   - sink/local/update-buffer 侧使用精确 qk; candidate 侧使用 RaBitQ 近似 qk。
   - 用 joint softmax 近似注意力权重: softmax(cat([sl_exact_qk, cand_approx_qk]) / sqrt(d))

   - benefit 指标定义为: benefit = attn_weight * ||v||_2
     其中 sink/local/update-buffer 的 value norm 来自 _sl_vnorm_cache, candidate 的 value norm 来自 value_norm_gpu。

   - 先固定保留 sink/local/update-buffer, 再对 candidate 按 benefit 降序排序，
     找到最小 k, 使得 sl_benefit + topk_cand_benefit >= threshold * total_benefit

   - 选 k 的指标和选 top-k indices 的排序指标保持一致，都是 value-weighted benefit。

   - 工程上为保持张量形状统一，最终实际 gather 使用 global_k=max(adaptive_k);

"""
import math
from pathlib import Path
from typing import Any
import torch
import torch.cuda.nvtx as nvtx
import numpy as np
import json
import os
import sys
import traceback
import time
from datetime import datetime
from termcolor import colored
import gc
# from retroinfer_kernels import ThreadPool, WaveBufferCPU
# from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter
from fast_hadamard_transform import hadamard_transform  # type: ignore
from cache_hub.rerank.rerank import partial_rerank_cuda

USE_RABITQ_RERANK = True    # True: RaBitQ 近似精排, False: 精确内积精排 (临时关闭验证 coarse recall)
DEFAULT_SRHT_ROUNDS = 1



# 导入基类
try:
    from .cache import KV_Cache
except ImportError:
    from cache import KV_Cache
from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat
# Steady Zone 动态更新的间隔已移至实例变量 self.dynamic_update_interval


# ==================== 内嵌 MultiDimBlockQuantizer 类 ====================
class MultiDimBlockQuantizer:
    """
    多维块结构化量化器(RxΩ分解)
    1. 将d维向量分成B个m维块
    2. 每个块分解为 radius x direction(极坐标形式)
    3. radius用解析Lloyd-Max量化,direction用球面CVT量化
    4. 支持ADC(Asymmetric Distance Computation)查询
    """
    
    def __init__(self, d, m, K_r, K_omega, codebook_path, seed=42, device=None):
        """
        初始化量化器(从预生成的码本加载到GPU)
        """
        assert d % m == 0, f"总维度{d}必须能被块维度{m}整除"
        if not torch.cuda.is_available():
            raise RuntimeError("GPU不可用!此版本只支持GPU模式")
    
        self.d = d
        self.m = m
        self.B = d // m
        self.K_r = K_r
        self.K_omega = K_omega
        self.K_total = K_r * K_omega
        self.seed = seed
                
        if device is None:
            device = torch.device('cuda:0')
        elif isinstance(device, str):
            device = torch.device(device)
        
        # 预生成 SRHT 对角符号矩阵 (GPU, bfloat16)
        # 保存当前随机状态，避免影响外部采样
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        torch.manual_seed(self.seed)  # 使用固定种子确保 SRHT 矩阵可重复
        self.srht_diagonal_signs = []
        for _ in range(DEFAULT_SRHT_ROUNDS):
            signs = torch.randint(0, 2, (self.d,), device=device, dtype=torch.bfloat16) * 2 - 1
            self.srht_diagonal_signs.append(signs)
        
        # 恢复之前的随机状态（不影响外部 token 采样）
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        
        # 加载码本到 GPU
        try:
            with open(codebook_path, 'r') as f:
                codebook = json.load(f)
            
            config = codebook['config']
            if (config['d'] != self.d or config['m'] != self.m or 
                config['K_r'] != self.K_r or config['K_omega'] != self.K_omega):
                raise ValueError(
                    f"码本参数不匹配!\n"
                    f"  期望: d={self.d}, m={self.m}, K_r={self.K_r}, K_omega={self.K_omega}\n"
                    f"  实际: d={config['d']}, m={config['m']}, K_r={config['K_r']}, K_omega={config['K_omega']}"
                )
            
            if "angular_signs" in codebook:
                angular_signs = torch.tensor(codebook["angular_signs"], dtype=torch.int8, device=device)
                angular_signs = torch.where(angular_signs >= 0,
                    torch.ones_like(angular_signs), -torch.ones_like(angular_signs)).to(torch.int8)
                # print("量化器初始化完成(K_r=1 优化:从符号码本加载)")
            else:
                centers = torch.tensor(codebook['angular_centers'], dtype=torch.bfloat16, device=device)
                angular_signs = torch.where(centers >= 0, 
                    torch.ones_like(centers), -torch.ones_like(centers)).to(torch.int8)
                # print("量化器初始化完成(K_r=1 优化:从 float 码本转换)")

            # 保存 bfloat16 版本（值只有 ±1）
            self.V_omega_gpu = angular_signs.to(torch.bfloat16)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"码本文件不存在: {codebook_path}")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[Init] 码本读取失败(Key/JSON): {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise ValueError(f"码本文件格式错误: {e}")


# ==================== 结束 MultiDimBlockQuantizer 类定义 ====================



class polar_cache(KV_Cache):

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        sink_size: int,
        local_size: int,
        core: int,
        nprobe: int,
        cache_unit_size: int,
        cache_cluster_num: int,
        num_gpus: int,
        model_size: int,
        codebook_path: str = '/data/qwang/q/thalia/kvcache/ANNCache/turboquant/codebooks/codebook_d128_m8_Kr1_Kw256_rabitq_sign.json',
        collision_ratio: float = 0.3,
        candidate_ratio: float = 0.05,
        final_topk: int = 100,  # 精排阶段：从候选集中选 top-K 个 key
        dynamic_update_interval: int = 512,  # 每累积多少个 tokens 触发更新
        full_attention_threshold: int = 2000,  # prefill + decode < threshold 时使用全量 attention (对齐 PQCache)
        enable_offload: bool = True,  # 消融实验：关闭 offload，key 都在 GPU
        enable_recall: bool = False,  # 接受 qwen_pai 透传，本类不读
        # ===== 自适应 Top-k 参数（每步都按 attention-mass 阈值确定 k） =====
        adaptive_topk_threshold: float = 0.9,  # 累积 attention mass 阈值
        adaptive_topk_max_k: int = 2048,  # per-head 最大 k，同时也是 UVA dst buffer 容量上限
    ) -> None:
        
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
         
        self.valid_start = valid_start  # 用于 prefill 时确定有效数据起始位置
        self.sink_size = sink_size  # Sink tokens 数量 = 4
        self.local_size = local_size   # Local window 大小 (通过配置文件设置,推荐512)
        self.steady_size = self.sink_size + self.local_size  # Steady Zone 总大小 = 4 + 512 = 516
        # 动态更新间隔：每累积多少个新 tokens 触发一次 local zone 更新
        self.dynamic_update_interval = dynamic_update_interval if dynamic_update_interval is not None else self.local_size
        self.group_size = self.num_heads // self.kv_head # 每个kv head对应的query head的group数量=32/8=4
        self.batch_groups = self.batch_size * self.kv_head # batch_size × kv_head = 1 × 8 = 8
        self.dtype = dtype
        # 保存 max_new_length,用于计算 retrieval zone capacity
        self.max_new_length = max_new_length
        # qwen.py 传入: max_length = real_input_length + max_new_length
        self.input_length = self.max_length - max_new_length  # 实际 prefill 长度 (通过反推得到)
        self.FULL_ATTENTION_THRESHOLD = full_attention_threshold  # prefill + decode < threshold 时全量计算
        self.current_seq_len = [0] * layer_num  # 每层独立跟踪，prefill 后会初始化为 input_length        
        self.codebook_path = codebook_path
        self.collision_ratio = collision_ratio
        self.candidate_ratio = candidate_ratio
        self.final_topk = final_topk  # 精排阶段的 top-K
        self.enable_offload = enable_offload  # 兼容性参数
        # 自适应 Top-k 参数
        self.adaptive_topk_threshold = adaptive_topk_threshold
        self.adaptive_topk_max_k = adaptive_topk_max_k if adaptive_topk_max_k is not None else final_topk
        try:
            with open(codebook_path, 'r') as f:
                codebook_config = json.load(f)['config']
            self.polar_m = codebook_config['m']
            self.polar_K_r = codebook_config['K_r']
            self.polar_K_omega = codebook_config['K_omega']
            self.polar_B = head_dim // self.polar_m  # 块数
            # print(f"  ✓ 从 codebook 自动读取参数: m={self.polar_m}, K_r={self.polar_K_r}, K_omega={self.polar_K_omega}")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"[Init] 读取 codebook 配置失败: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise ValueError(f"无法从 codebook 读取参数: {e}")
        # 初始化实际使用的 retrieval zone 长度
        self._actual_retrieval_length = None  # 将在 prefill 时设置
        # 添加decode步数计数器,用于控制Recall计算频率
        self._decode_step_counter = 0
        # 上次 buffer reset 时的 counter 值（长 prefill offload 用）
        self._buffer_base_counter = 0
        # 延迟 base 更新标志（确保 update 后的第一个 token 写到 offset 0）
        self._pending_buffer_reset = False
        

        # 初始化Polar ANN量化器(共享一个,码本数据无关)
        import time as _time
        _t0 = _time.time()
        print(f"[Init] polar_cache.__init__ start (input_length={self.input_length}, max_new={max_new_length}, layers={layer_num}, offload={self.enable_offload})", flush=True)
        self.device = torch.device(list[Any](layer_mapping.values())[0] if layer_mapping else 'cuda:0')
        self.quantizer = MultiDimBlockQuantizer(
            d=head_dim,
            m=self.polar_m,
            K_r=self.polar_K_r,
            K_omega=self.polar_K_omega,
            seed=42,
            codebook_path=codebook_path,
            device=self.device
        )

        # ============ 符号码本（用于 coarse retrieval）============
        # V_omega_gpu 是 ±1 的符号码本 [K_omega, m]
        self._V_omega_gpu_T = self.quantizer.V_omega_gpu.T.contiguous()  # [m, K_omega] 预转置
        # 统一 KV Cache 架构（按生成顺序存储所有 tokens）: 全量容量 = prefill + decode 的总长度
        self.unified_kv_capacity = self.input_length + self.max_new_length
        # ============ 4-bit 幅值量化参数（1-bit符号 + 3-bit幅值 = 8 levels）============
        # 由 Lloyd-Max 从球面均匀分布采样得到，对任何数据集通用 生成脚本: python run/generate_magnitude_levels.py --levels 8
        self._mag_thresholds = torch.tensor(
            [0.0843, 0.1698, 0.2578, 0.3499, 0.4487, 0.5585, 0.6901],
            device=self.device, dtype=torch.bfloat16).contiguous()  # [7] 决策阈值（bf16，~2-3位小数精度）
        self._mag_centers = torch.tensor([0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
            device=self.device, dtype=torch.bfloat16).contiguous()  # [8] 重构中心 a[t]
        # ============ 预缓存 shifts 张量（避免每次调用时创建）============
        # bitpack 符号用：[1, 2, 4, 8, 16, 32, 64, 128]
        self._bitpack_shifts = (1 << torch.arange(self.polar_m, device=self.device, dtype=torch.uint8))
        print(f"[Init] quantizer + codebook ready ({_time.time()-_t0:.1f}s)", flush=True)

        
        if self.enable_offload:
            # 【Offload 模式】GPU 只存 Sink + Local + Update Buffer（小容量，省显存）
            #  Retrieval Zone 的 KV offload 到 CPU（大容量，pinned memory）
            self.gpu_kv_capacity = self.sink_size + self.local_size + self.dynamic_update_interval
            print(f"[Init] Offload Mode: ENABLED (gpu_kv={self.gpu_kv_capacity}, cpu_kv={self.unified_kv_capacity - self.sink_size - self.local_size})", flush=True)
            
            # 创建专用 copy stream，实现 GPU→CPU 拷贝与 Attention 计算并行
            self._copy_stream = torch.cuda.Stream(device=self.device)
            # - main_event: 标记源数据准备好（default stream → copy stream 的依赖）
            # - copy_event: 标记该层 D2H 拷贝完成（CPU 数据可安全读取）
            self._main_events = [torch.cuda.Event() for _ in range(layer_num)]
            self._copy_events = [torch.cuda.Event() for _ in range(layer_num)]
            # 让 copy_event 初始处于“已完成”状态，避免第一次 synchronize() 卡住
            with torch.cuda.device(self.device):
                for e in self._copy_events:
                    e.record()

            # 保存仍在 D2H 拷贝中的源 tensor 引用，避免 allocator 复用导致潜在数据竞争
            # 每层一个 (keys, values) tuple；拷贝完成后会清理
            self._inflight_offload_buffers = [None for _ in range(layer_num)]
            self._layer_offload_inflight = [False for _ in range(layer_num)]
            # CPU 只存 Retrieval Zone，不含 Sink + Local
            self.cpu_kv_capacity = self.unified_kv_capacity - self.sink_size - self.local_size
            # print(f"  ✓ GPU KV Capacity: {self.gpu_kv_capacity} tokens (sink={self.sink_size} + local={self.local_size} + update_buffer={self.dynamic_update_interval})")
            # print(f"  ✓ CPU KV Capacity: {self.cpu_kv_capacity} tokens (pinned memory for retrieval zone)")
            
            # GPU 端：小容量，只存 Sink + Local + Update Buffer
            print(f"[Init] allocating GPU KV ({layer_num} layers × {self.gpu_kv_capacity} tokens) ...", flush=True)
            self.unified_keys_gpu = []
            self.unified_values_gpu = []
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
            print(f"[Init] GPU KV done ({_time.time()-_t0:.1f}s)", flush=True)
            
            # CPU 端：只存 Retrieval Zone 的 KV（pinned memory 加速传输）
            # 分配前先清理内存碎片
            gc.collect()
            torch.cuda.empty_cache()
            
            self.unified_keys_cpu = []
            self.unified_values_cpu = []
            
            print(f"[Init] allocating CPU pinned KV ({layer_num} layers × {self.cpu_kv_capacity} tokens) ...", flush=True)
            for ldx in range(layer_num):
                # 每层分配前清理，帮助整理内存碎片
                if ldx % 7 == 0:
                    gc.collect()
                self.unified_keys_cpu.append(torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim), dtype=dtype, pin_memory=True))
                self.unified_values_cpu.append(torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim), dtype=dtype, pin_memory=True))
            print(f"[Init] CPU pinned KV done ({_time.time()-_t0:.1f}s)", flush=True)
                
            # 记录 CPU 上已存储的有效范围
            self.cpu_valid_end = [0] * layer_num

            # 预分配 TopK gather 结果的 GPU buffer，给 UVA h2d_gather_kv 当 dst 用
            # 上界用 adaptive_topk_max_k：每步 global_k = max(adaptive_k) <= max_k
            # 避免每次 decode 都新分配显存；buffer 写满前 global_k 行才有效，concat 时切片
            _topk_buf_cap = self.adaptive_topk_max_k
            print(f"[Init] allocating TopK gather GPU dst buffer (cap={_topk_buf_cap}) ...", flush=True)
            self._topk_keys_buffer = torch.empty(
                (batch_size, self.kv_head, _topk_buf_cap, head_dim),
                dtype=dtype, device=self.device,
            ).contiguous()
            self._topk_values_buffer = torch.empty(
                (batch_size, self.kv_head, _topk_buf_cap, head_dim),
                dtype=dtype, device=self.device,
            ).contiguous()

            print(f"[Init] allocating offload staging buffers ...", flush=True)
            # 预分配 Offload staging buffers（避免 clone + 避免跨层复用导致数据竞争）
            # 每层一个 staging buffer：否则在 _update_polar_index 的 layer 循环里复用同一个 buffer，
            # 会和异步 copy_stream 产生 race（上一层还在 D2H，下一层就覆盖 src）。
            self._offload_keys_buffer = [
                torch.empty(
                    (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                    dtype=dtype, device=self.device
                )
                for _ in range(layer_num)
            ]
            self._offload_values_buffer = [
                torch.empty(
                    (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                    dtype=dtype, device=self.device
                )
                for _ in range(layer_num)
            ]

        else:
            # ===========================================
            # 【非 Offload 模式】GPU 存全量
            self.gpu_kv_capacity = self.unified_kv_capacity
            print(f"[Init] No-Offload Mode: GPU full capacity = {self.unified_kv_capacity} tokens", flush=True)
            print(f"[Init] allocating GPU KV ({layer_num} layers × {self.unified_kv_capacity} tokens) ...", flush=True)
            self.unified_keys_gpu = []
            self.unified_values_gpu = []
            
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.unified_kv_capacity, head_dim), dtype=dtype,
                        device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.unified_kv_capacity, head_dim), dtype=dtype,
                        device=self.device).contiguous())
                if ldx % 10 == 0:
                    print(f"[Init]   GPU KV layer {ldx}/{layer_num} ({_time.time()-_t0:.1f}s)", flush=True)
            print(f"[Init] GPU KV done ({_time.time()-_t0:.1f}s)", flush=True)
            
            self.unified_keys_cpu = None
            self.unified_values_cpu = None
            self.cpu_valid_end = None
            self._topk_keys_buffer = None
            self._topk_values_buffer = None
        
        # 注意：在统一架构下，retrieval_zone_capacity 指的是 unified_kv_capacity
        self.retrieval_zone_capacity = self.unified_kv_capacity
        # Incremental value norm cache for sink+local+buffer (避免每步重算 L2 norm)
        self._sl_vnorm_cache = [
            torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity),
                        dtype=torch.float32, device=self.device)
            for _ in range(layer_num)
        ]
        print(f"[Init] allocating codebook/weight/4bit ({layer_num} layers × {self.retrieval_zone_capacity} tokens) ...", flush=True)
        # RaBitQ rerank 预计算权重（已融合 key_norm）：
        # weight_{k,b} = (r_{k,b} / alpha_{k,b}) * ||k||  shape: [batch, kv_head, B, N] - 与 codebook 对齐
        self.key_block_weight_gpu = []
        for ldx in range(layer_num):
            self.key_block_weight_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity),
                    dtype=torch.bfloat16,device=self.device,).contiguous())
        
        # ============ 4-bit packed 存储（1-bit sign + 3-bit magnitude）============
        # 每个 block（m=8 维）4-bit/维 × m维 = 32 bits = 4 bytes
        # shape: [batch, kv_head, B, N, bytes_per_block]
        self._4bit_bytes_per_block = self.polar_m // 2  # m=8 -> 4 bytes
        self.key_4bit_packed_gpu = []
        for ldx in range(layer_num):
            self.key_4bit_packed_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity, self._4bit_bytes_per_block),
                    dtype=torch.uint8,device=self.device).contiguous())

        # Codebook存储(GPU): [layer][batch, kv_head, B, kv_len]
        self.codebook_gpu = [torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity),dtype=torch.uint8, device=self.device) for _ in range(layer_num)]
        # cluster_key_counts: [layer][batch, kv_head, B, K_omega] - 每个cluster的key数量
        self.cluster_key_counts_gpu = [torch.zeros((batch_size, self.kv_head, self.polar_B, self.polar_K_omega), dtype=torch.int32, device=self.device) for _ in range(layer_num)]  
        # value L2 norm: [layer][bs, kv_heads, retrieval_zone_capacity] - 用于 value-weighted adaptive-k
        self.value_norm_gpu = [
            torch.zeros((batch_size, self.kv_head, self.retrieval_zone_capacity),
                        dtype=torch.float32, device=self.device)
            for _ in range(layer_num)
        ]
        # 调试用：本轮 update flush 之间每一层 per-head 的最大 adaptive_k 快照
        self._adaptive_k_log_max = [None] * layer_num
        self.context = 0
        self._hadamard_scale = 1.0 / np.sqrt(self.head_dim)
        self._attn_scale = 1.0 / np.sqrt(self.head_dim)
        print(f"[Init] codebook/weight/4bit done ({_time.time()-_t0:.1f}s)", flush=True)
 
        
        
        
        # 设置 PyTorch CPU 线程数（只在第一次初始化时设置）
        if not hasattr(polar_cache, '_threads_initialized'):
            torch.set_num_threads(1)          # 8 关闭多线程，单线程运行
            try:
                torch.set_num_interop_threads(1)  # inter-op 也用 1
            except RuntimeError:
                pass  # 已经设置过或并行工作已启动，忽略
            polar_cache._threads_initialized = True
            # print(f"  [Thread Config] intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}")
        
        # 预加载 C++ gather extension（避免 decode 循环内重复检查）
        print(f"[Init] loading CUDA extensions ...", flush=True)
        try:
            from .gather import load_gather_ext
            self._gather_ext = load_gather_ext()
            print(f"[Init]   gather ext loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   gather ext FAILED: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self._gather_ext = None
        
        # 预加载 Radix TopK CUDA extension (O(n) vs O(n log k))
        try:
            from .topk import load_radix_topk_ext
            self._radix_topk_ext = load_radix_topk_ext()
            print(f"[Init]   radix_topk loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   radix_topk FAILED: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self._radix_topk_ext = None
        
        # 预加载融合版本的 Collision CUDA kernel（预编译，避免首次调用时卡住）
        try:
            from .collision_fused.collison_interface import update_cache_cnt_cuda_interface, load_kernel_module as load_collision_module
            self._update_cache_cnt_fused = update_cache_cnt_cuda_interface
            load_collision_module("collision.cu", "update_cache_cnt")
            print(f"[Init]   collision_fused loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   collision_fused FAILED: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"Collision Fused CUDA kernel compile failed: {e}")
        
        # 预加载 UVA H2D Gather kernel（利用 UVA 直接从 pinned CPU gather 到 GPU）
        try:
            from .gather_trans.trans_h2d import h2d_gather_kv
            self._h2d_gather_kv = h2d_gather_kv
            print(f"[Init]   h2d_gather loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   h2d_gather FAILED: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"H2D Gather CUDA kernel compile failed: {e}")
        
        # 预加载 Rerank CUDA extension (partial_rerank)
        try:
            from .rerank.rerank import load_kernel_module
            self._rerank_kernel = load_kernel_module("rerank.cu", "rerank")
            print(f"[Init]   rerank loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   rerank FAILED: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"Rerank CUDA kernel compile failed: {e}")
        
        # __init__ 完成，清理初始化过程中的临时对象
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Init] polar_cache.__init__ DONE ({_time.time()-_t0:.1f}s)", flush=True)


    # 处理单层prefill阶段（单样本）
    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx):
        """
        Prefill 阶段入口：
        - 兼容旧逻辑: bsz==1 时按 batch_idx 单样本处理
        - lockstep multi-batch: bsz>1 时一次性处理整个 batch(要求各样本有效长度一致)
        """
        bsz = key_states.shape[0]
        if bsz == 1:
            return self._prefill_single_sample(key_states, value_states, layer_idx, batch_idx)
        # 对 bsz>1：只在 batch_idx==0 时执行一次，避免外部仍按 batch_idx 循环时重复工作
        if batch_idx != 0:
            # 保持返回 shape 兼容：去掉 padding（按 batch_idx=0 的 valid_start 规则）
            vs = self.valid_start[0] if isinstance(self.valid_start, (list, tuple)) else int(self.valid_start)
            return key_states[:, vs:, :, :], value_states[:, vs:, :, :]
        return self._prefill_lockstep_batch(key_states, value_states, layer_idx)


    def _prefill_lockstep_batch(self, key_states, value_states, layer_idx: int):
        """
        lockstep multi-batch prefill：
        - 假设所有 batch 的有效 token 长度一致（典型 benchmark/padding 场景）
        - 当前实现要求 valid_start 在 batch 内一致（否则需要 per-batch region，改动更大）
        """
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == self.batch_size, f"bsz({bsz}) must equal cache batch_size({self.batch_size}) for lockstep mode."
        assert seq_len <= self.input_length

        # valid_start：要求 batch 内一致（支持 list/tuple/numpy array）
        if hasattr(self.valid_start, '__len__') and len(self.valid_start) > 1:
            assert len(self.valid_start) >= bsz
            vs0 = int(self.valid_start[0])
            for b in range(1, bsz):
                assert int(self.valid_start[b]) == vs0, (
                    f"lockstep multi-batch currently requires same valid_start across batch. "
                    f"valid_start[0]={vs0}, valid_start[{b}]={self.valid_start[b]}"
                )
            valid_start = vs0
        else:
            valid_start = int(self.valid_start) if not hasattr(self.valid_start, '__len__') else int(self.valid_start[0])

        # 区域划分（与单样本一致，只是对整个 batch 一次性 copy/encode）
        sink_start = valid_start
        sink_end = valid_start + self.sink_size
        retrieval_start = sink_end
        retrieval_end_raw = seq_len - self.local_size
        retrieval_end = max(retrieval_start, retrieval_end_raw)
        local_start = retrieval_end
        local_end = seq_len
        retrieval_length = retrieval_end - retrieval_start
        valid_length = retrieval_length



        if self._actual_retrieval_length is None:
            self._actual_retrieval_length = valid_length

        # 短 prefill：全部留在 GPU（不触发 retrieval/offload）
        if valid_length == 0:
            if layer_idx == 0 and not hasattr(self, '_short_prefill_printed'):
                # print(f"[Short Prefill] prefill_length={seq_len} <= steady_size={self.steady_size}, no retrieval zone")
                self._short_prefill_printed = True
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            k_gpu = key_states[:, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            v_gpu = value_states[:, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :actual_len, :] = k_gpu
            self.unified_values_gpu[layer_idx][:bsz, :, :actual_len, :] = v_gpu
            self._sl_vnorm_cache[layer_idx][:bsz, :, :actual_len] = torch.linalg.vector_norm(
                v_gpu, ord=2, dim=-1).to(torch.float32)
            self.current_seq_len[layer_idx] = actual_len
            if layer_idx == self.layer_num - 1:
                # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        # Step1: 分区存储 KV
        local_len = local_end - local_start
        if self.enable_offload:
            # Sink → GPU[:sink_size]
            sink_k = key_states[:, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
            sink_v = value_states[:, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :self.sink_size, :] = sink_k
            self.unified_values_gpu[layer_idx][:bsz, :, :self.sink_size, :] = sink_v

            # Local → GPU[sink_size : sink_size+local_len]
            local_k = key_states[:, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
            local_v = value_states[:, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, self.sink_size:self.sink_size+local_len, :] = local_k
            self.unified_values_gpu[layer_idx][:bsz, :, self.sink_size:self.sink_size+local_len, :] = local_v

            sl_end = self.sink_size + local_len
            self._sl_vnorm_cache[layer_idx][:bsz, :, :self.sink_size] = torch.linalg.vector_norm(
                sink_v, ord=2, dim=-1).to(torch.float32)
            self._sl_vnorm_cache[layer_idx][:bsz, :, self.sink_size:sl_end] = torch.linalg.vector_norm(
                local_v, ord=2, dim=-1).to(torch.float32)

            # Retrieval → CPU（异步 D2H）
            temp_retrieval_keys = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            temp_retrieval_values = value_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
            self._layer_offload_inflight[layer_idx] = True
            self._main_events[layer_idx].record()
            
            with torch.cuda.stream(self._copy_stream):
                self._main_events[layer_idx].wait()
                self.unified_keys_cpu[layer_idx][:bsz, :, :retrieval_length, :].copy_(temp_retrieval_keys, non_blocking=True)
                self.unified_values_cpu[layer_idx][:bsz, :, :retrieval_length, :].copy_(temp_retrieval_values, non_blocking=True)
                self._copy_events[layer_idx].record()
            self.cpu_valid_end[layer_idx] = retrieval_length

        else:
            # 非 Offload：全部存 GPU
            actual_len = seq_len - valid_start
            k_gpu = key_states[:, valid_start:seq_len, :, :].transpose(1, 2).to(self.dtype)
            v_gpu = value_states[:, valid_start:seq_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :actual_len, :] = k_gpu
            self.unified_values_gpu[layer_idx][:bsz, :, :actual_len, :] = v_gpu
            self._sl_vnorm_cache[layer_idx][:bsz, :, :actual_len] = torch.linalg.vector_norm(
                v_gpu, ord=2, dim=-1).to(torch.float32)

        self.current_seq_len[layer_idx] = seq_len - valid_start

        # Step2: encode retrieval zone（一次性 batch_encode）
        keys_to_encode = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).contiguous()
        codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(keys_to_encode, layer_idx=layer_idx, return_rabitq=USE_RABITQ_RERANK)
        
        if layer_idx == 0:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            # print(colored(f"[MEM DEBUG] batch_encode 完成后: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB", 'yellow'))
        
        self.codebook_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end] = codebook_batch
        if USE_RABITQ_RERANK:
            self.key_block_weight_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end] = block_weight_batch
            self.key_4bit_packed_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end, :] = packed4b_batch

        # Step3: cluster counts（一次性处理整个 batch）
        self.cluster_key_counts_gpu[layer_idx][:bsz] = self._count_cluster_keys(codebook_batch, codebook_batch.device)

        # Step4: retrieval zone values 的 L2 norm（用于 value-weighted adaptive-k）
        # 注：与 _prefill_single_sample 同一索引约定 —— value_norm_gpu 用 retrieval_start 偏移
        values_retrieval = value_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2)  # [bs, kv_heads, ret_len, d]
        v_norm = torch.linalg.vector_norm(values_retrieval, ord=2, dim=-1)  # [bs, kv_heads, ret_len]
        self.value_norm_gpu[layer_idx][:bsz, :, retrieval_start:retrieval_end] = v_norm.to(torch.float32)

        if layer_idx == self.layer_num - 1:
            # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
            self.context += self.current_seq_len[layer_idx]
            if self.enable_offload:
                self._sync_offload_copies()
            # 确保所有 GPU 操作完成（避免异步错误延迟报告）
        else:
            if self.enable_offload:
                self._sync_layer_offload(layer_idx)

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]


    def _prefill_single_sample(self, key_states, value_states, layer_idx, batch_idx):
        """
        Prefill阶段:使用Polar ANN编码keys(完整GPU版本)
        主要改动:
        1. 不用K-means,用encode()量化keys(在GPU上进行)
        2. 构建倒排索引而非centroids(在GPU上进行)
        3. 所有数据保持在GPU上,无 CPU offload
        """
        bsz, seq_len, group_num, head_dim = key_states.shape        
        assert bsz == 1, "Multi-batch prefilling only support single batch one by one."
        assert seq_len <= self.input_length        
        valid_start = self.valid_start[batch_idx]  
        # Sink Zone: [valid_start, sink_end)
        sink_start = valid_start # 0
        sink_end = valid_start + self.sink_size   # 4
        # 注意:当 seq_len < steady_size 时,会出现以下情况:
        #    - retrieval_end 会是负数(因为 seq_len - local_size < 0) 这时应该理解为:没有 Retrieval Zone, 所有数据都在 Steady Zone
        #    - Retrieval Zone: [retrieval_start, retrieval_end)
        retrieval_start = sink_end  # 4
        retrieval_end_raw = seq_len - self.local_size  # 139-512=-373可能为负数
        retrieval_end = max(retrieval_start, retrieval_end_raw)  # 4 确保 >= retrieval_start
        # Local Zone: [local_start, local_end)
        local_start = retrieval_end  # 4 如果 retrieval_end = retrieval_start,说明没有 retrieval zone
        local_end = seq_len  # 139
        retrieval_length = retrieval_end - retrieval_start  # 现在一定 >= 0
        valid_length = retrieval_length  # 保持变量名兼容性
        
        # 设置初始的 retrieval zone 长度
        if self._actual_retrieval_length is None:
            self._actual_retrieval_length = valid_length
            
        # 如果 retrieval_length = 0,说明 prefill 长度太短, kvcache全部保存在 GPU(Steady Zone)中
        if valid_length == 0:
            if layer_idx == 0 and not hasattr(self, '_short_prefill_printed'):
                # print(f"[Short Prefill] prefill_length={seq_len} <= steady_size={self.steady_size}, no retrieval zone")
                self._short_prefill_printed = True
            # 短序列全部放 GPU（不超过 gpu_kv_capacity）
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            self.unified_keys_gpu[layer_idx][batch_idx, :, :actual_len, :] = key_states[0, valid_start:valid_start+actual_len, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :actual_len, :] = value_states[0, valid_start:valid_start+actual_len, :, :].transpose(0, 1).to(self.dtype)
            self._sl_vnorm_cache[layer_idx][batch_idx:batch_idx+1, :, :actual_len] = torch.linalg.vector_norm(
                self.unified_values_gpu[layer_idx][batch_idx:batch_idx+1, :, :actual_len, :],
                ord=2, dim=-1).to(torch.float32)
            self.current_seq_len[layer_idx] = actual_len
            if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]
        
        # ===========================================
        # Step1: 将 prefill tokens 分区存储
        # Offload 模式：Sink+Local → GPU, Retrieval → CPU
        # 非 Offload 模式：全部 → GPU
        local_len = local_end - local_start
        
        if self.enable_offload:
            # 【Offload 模式】GPU 只存 Sink + Local → GPU(Sink Zone + Local Zone)
            self.unified_keys_gpu[layer_idx][batch_idx, :, :self.sink_size, :] = key_states[0, sink_start:sink_end, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :self.sink_size, :] = value_states[0, sink_start:sink_end, :, :].transpose(0, 1).to(self.dtype)
            
            # Local → GPU[sink_size:sink_size+local_len]
            self.unified_keys_gpu[layer_idx][batch_idx, :, self.sink_size:self.sink_size+local_len, :] = key_states[0, local_start:local_end, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, self.sink_size:self.sink_size+local_len, :] =  value_states[0, local_start:local_end, :, :].transpose(0, 1).to(self.dtype)

            sl_end = self.sink_size + local_len
            self._sl_vnorm_cache[layer_idx][batch_idx:batch_idx+1, :, :self.sink_size] = torch.linalg.vector_norm(
                self.unified_values_gpu[layer_idx][batch_idx:batch_idx+1, :, :self.sink_size, :],
                ord=2, dim=-1).to(torch.float32)
            self._sl_vnorm_cache[layer_idx][batch_idx:batch_idx+1, :, self.sink_size:sl_end] = torch.linalg.vector_norm(
                self.unified_values_gpu[layer_idx][batch_idx:batch_idx+1, :, self.sink_size:sl_end, :],
                ord=2, dim=-1).to(torch.float32)

            # Retrieval Zone → CPU（异步拷贝）
            temp_retrieval_keys = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).to(self.dtype)
            temp_retrieval_values = value_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).to(self.dtype)
            # 保留引用：避免异步拷贝过程中源 tensor 被 allocator 复用
            self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
            self._layer_offload_inflight[layer_idx] = True
            # 主流记录 event：数据已准备好
            self._main_events[layer_idx].record()
            
            # Copy stream 异步拷贝：不阻塞主流，与后续 batch_encode 并行
            with torch.cuda.stream(self._copy_stream):
                self._main_events[layer_idx].wait()  # 等待数据准备好
                # 使用 copy_ + non_blocking 做异步拷贝（目标是 pinned memory）
                self.unified_keys_cpu[layer_idx][batch_idx, :, :retrieval_length, :].copy_(temp_retrieval_keys, non_blocking=True)
                self.unified_values_cpu[layer_idx][batch_idx, :, :retrieval_length, :].copy_(temp_retrieval_values, non_blocking=True)
                self._copy_events[layer_idx].record()  # 标记该层拷贝完成
            
            self.cpu_valid_end[layer_idx] = retrieval_length
            
            # if layer_idx == 0 and batch_idx == 0:
                # print(colored(f"[Offload Prefill] Sink({self.sink_size})+Local({local_len})→GPU, Retrieval({retrieval_length})→CPU (async)", 'yellow'))
                # pass
        else:
            # 【非 Offload 模式】全部存 GPU
            actual_len = seq_len - valid_start  # 有效 token 数（去除左侧 padding）
            self.unified_keys_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                key_states[0, valid_start:seq_len, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                value_states[0, valid_start:seq_len, :, :].transpose(0, 1).to(self.dtype)
            self._sl_vnorm_cache[layer_idx][batch_idx:batch_idx+1, :, :actual_len] = torch.linalg.vector_norm(
                self.unified_values_gpu[layer_idx][batch_idx:batch_idx+1, :, :actual_len, :],
                ord=2, dim=-1).to(torch.float32)

        # 记录有效序列长度（用于 decode 时确定写入位置）
        self.current_seq_len[layer_idx] = seq_len - valid_start
        

        #  step2: 计算方向码本(量化keys,K_r=1 优化:不再计算半径码) 提取 Retrieval Zone 用于编码（仅用于构建索引，不需要单独存储）
        # ----- Step2a: batch_encode -----
        # nvtx.range_push("prefill_step1_batch_encode")
        # key_states: [1, seq_len, kv_head, head_dim] -> [1, kv_head, retrieval_length, head_dim]
        keys_to_encode = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).contiguous().unsqueeze(0)
        codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
            keys_to_encode, layer_idx=layer_idx, return_rabitq=USE_RABITQ_RERANK)
        
        # codebook_batch: [1, kv_head, B, retrieval_length]（batch_encode 已转置）
        self.codebook_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end] = codebook_batch[0]
        if USE_RABITQ_RERANK:
            # block_weight_batch: [1, kv_head, B, retrieval_length] bf16
            # packed4b_batch: [1, kv_head, B, retrieval_length, 4]（batch_encode 已转置）
            self.key_block_weight_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end] = block_weight_batch[0]
            self.key_4bit_packed_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end, :] = packed4b_batch[0]
        # nvtx.range_pop()
        
        # step3: 统计每个cluster包含的key数量（codebook_batch 是 [1, kv_heads, B, len]，取 [0]）
        # nvtx.range_push("prefill_step3_cluster_count")
        self.cluster_key_counts_gpu[layer_idx][batch_idx] = self._count_cluster_keys(codebook_batch, codebook_batch.device)[0]  # [1, kv_heads, B, K_omega] -> [kv_heads, B, K_omega]
        # nvtx.range_pop()

        # step4: 计算 retrieval zone values 的 L2 norm（用于 value-weighted adaptive-k）
        values_retrieval = value_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1)  # [kv_heads, ret_len, d]
        v_norm = torch.linalg.vector_norm(values_retrieval, ord=2, dim=-1)  # [kv_heads, ret_len]
        self.value_norm_gpu[layer_idx][batch_idx, :, retrieval_start:retrieval_end] = v_norm.to(torch.float32)
        
        
        # 更新context:目前处理的tokens数量（有效长度，不含 padding）
        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += self.current_seq_len[layer_idx]  # 已在上面计算
            # 【同步点】Prefill 结束，等待所有异步 offload 完成
            if self.enable_offload:
                self._sync_offload_copies()
        else:
            # Prefill 的 key_states/value_states 往往是短生命周期张量（来自当前 layer forward）。
            # 为避免“跨层累计持有大张量引用”导致峰值显存暴涨，这里按层等待本层 D2H 完成并清理引用。
            if self.enable_offload:
                self._sync_layer_offload(layer_idx)
                
        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]
    
    
    def _sync_offload_copies(self):
        """等待所有异步 offload 拷贝完成"""
        if not self.enable_offload:
            return
        # 等待 copy stream 上的所有操作完成
        self._copy_stream.synchronize()
        # 清理临时引用
        if hasattr(self, "_inflight_offload_buffers"):
            for ldx in range(self.layer_num):
                self._inflight_offload_buffers[ldx] = None
                self._layer_offload_inflight[ldx] = False


    def _sync_layer_offload(self, layer_idx: int):
        """只等待指定层的异步 offload 完成（PQCache 风格：按层精确等待）。"""
        if not self.enable_offload:
            return
        if not hasattr(self, "_copy_events"):
            return
        if hasattr(self, "_layer_offload_inflight") and (not self._layer_offload_inflight[layer_idx]):
            return
        self._copy_events[layer_idx].synchronize()
        if hasattr(self, "_inflight_offload_buffers"):
            self._inflight_offload_buffers[layer_idx] = None
        if hasattr(self, "_layer_offload_inflight"):
            self._layer_offload_inflight[layer_idx] = False


    def _compute_local_start(self, layer_idx=0):
        """
        轻量版：只计算 local_start，不构建 tensor（用于 _update_polar_index）
        """
        total_len = self.current_seq_len[layer_idx]
        steady_threshold = self.steady_size  # 预计算：sink_size + local_size
        
        if self.input_length >= steady_threshold:
            update_buffer_size = max(0, total_len - self.input_length) % self.dynamic_update_interval
        elif total_len >= steady_threshold:
            update_buffer_size = (total_len - steady_threshold) % self.dynamic_update_interval
        else:
            update_buffer_size = 0
        local_end = max(total_len - update_buffer_size, self.sink_size)
        return max(self.sink_size, local_end - self.local_size)
    
    
    
    
    def _get_full_kv_for_attention(self, layer_idx, sink_indices, local_indices, 
                                    retrieval_start, retrieval_length, device, dtype):
        """
        获取完整的 KV 用于 attention 计算（Sink + Retrieval + Local）
        统一处理 Offload 和非 Offload 模式，避免代码重复
        
        Args:
            layer_idx: 层索引
            sink_indices: Sink 区域的全局索引
            local_indices: Local+Buffer 区域的全局索引  
            retrieval_start: Retrieval 区域起始位置
            retrieval_length: Retrieval 区域长度（理论值）
            device: 目标设备
            dtype: 数据类型
        Returns:
            (concat_keys, concat_values): 拼接后的 KV，形状 [bs, kv_head, total_len, head_dim]
        """
        local_len = local_indices.shape[0]
        
        if self.enable_offload and self.unified_keys_cpu is not None and retrieval_length > 0:
            self._sync_layer_offload(layer_idx)
            
            # Sink+Local 在 GPU 上（先提取出来，保证变量作用域）
            sink_local_end = self.sink_size + local_len
            sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
            sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]
            
            cpu_retrieval_len = self.cpu_valid_end[layer_idx]
            if cpu_retrieval_len > 0:
                # Retrieval 数据已 offload 到 CPU，拼接: Sink+Local + Retrieval（causal=False，顺序无关）
                retrieval_keys = self.unified_keys_cpu[layer_idx][:, :, :cpu_retrieval_len, :].to(device, non_blocking=True)
                retrieval_values = self.unified_values_cpu[layer_idx][:, :, :cpu_retrieval_len, :].to(device, non_blocking=True)
                concat_keys = torch.cat([sink_local_keys, retrieval_keys], dim=2)
                concat_values = torch.cat([sink_local_values, retrieval_values], dim=2)
            else:
                # CPU 没有 offload 数据，所有 token 仍在 GPU，直接用 Sink+Local
                concat_keys = sink_local_keys
                concat_values = sink_local_values
        
        else:
            # 【非 Offload 模式】全部从 GPU 获取
            if retrieval_length > 0:
                retrieval_end = retrieval_start + retrieval_length
                retrieval_indices = torch.arange(retrieval_start, retrieval_end, device=device, dtype=torch.long)
                # 按物理顺序拼接：Sink → Retrieval → Local
                all_indices = torch.cat([sink_indices, retrieval_indices, local_indices])
                concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
                concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)
            else:
                # 没有 retrieval zone，只用 Sink + Local
                all_indices = torch.cat([sink_indices, local_indices]) if local_len > 0 else sink_indices
                concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
                concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)
        
        return concat_keys, concat_values



    def _update_polar_index(self):
        """
        动态更新:对Steady Zone中旧的Local部分进行Polar编码并更新倒排索引(GPU版本)
        逻辑:
        1. 提取旧的Local tokens [4:68] → 编码 → 追加到统一KV缓存
        2. 把新生成的64个tokens [68:132] 移到Local位置 [4:68]
        3. Steady Zone重置为 [Sink(4) + Local(64)] = 68
        """
        # 计算需要编码的范围：[old_retrieval_end : new_retrieval_end]
        # 关键：这里用的是 _update_polar_index 被调用时的 current_seq_len[0]
        # 这个值是在写入当前 token 之前的值！
        new_retrieval_end = self._compute_local_start(0)
        old_retrieval_end = self._actual_retrieval_length + self.sink_size
        encode_start = old_retrieval_end
        encode_end = new_retrieval_end
        num_new_tokens = encode_end - encode_start
        
        # 边界检查（触发时序已修复，num_new_tokens 应该总是 > 0）
        assert num_new_tokens > 0, f"num_new_tokens={num_new_tokens} <= 0, 触发逻辑有 bug"
        assert encode_start >= self.sink_size, f"encode_start ({encode_start}) < sink_size ({self.sink_size})"
        assert encode_end <= self.current_seq_len[0], f"encode_end ({encode_end}) > current_seq_len ({self.current_seq_len[0]})"

        # 更新 _actual_retrieval_length（编码到的相对长度）
        # Retrieval Zone = [sink_size : encode_end]
        self._actual_retrieval_length = encode_end - self.sink_size
        
        for ldx in range(self.layer_num):
            # 获取需要编码的新增部分
            if self.enable_offload:
                # 【Offload 模式】从 GPU 获取即将 offload 的 tokens
                # GPU 布局: [sink | local | update_buffer]
                # 需要编码的是 [sink : sink + num_new_tokens]（旧 Local + Buffer 前半）
                offload_start = self.sink_size
                offload_end = self.sink_size + num_new_tokens
                new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, offload_start:offload_end, :]
            else:
                # 【非 Offload 模式】按全局位置获取
                new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, encode_start:encode_end, :]
                
            # 统一架构：批量编码新增部分，按绝对位置存储
            # 注意：codebook/weight/4bit 已预分配 input_length + max_new_length 容量
            # 一次性处理整个 batch（避免逐 batch 循环的性能开销）
            # new_tokens_to_encode: [batch_size, kv_heads, num_new_tokens, head_dim]
            codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
                new_tokens_to_encode, layer_idx=ldx, return_rabitq=USE_RABITQ_RERANK)
            # 输出: [batch_size, kv_heads, B, num_new_tokens, ...]
            self.codebook_gpu[ldx][:, :, :, encode_start:encode_end] = codebook_batch
            if USE_RABITQ_RERANK:
                self.key_block_weight_gpu[ldx][:, :, :, encode_start:encode_end] = block_weight_batch
                self.key_4bit_packed_gpu[ldx][:, :, :, encode_start:encode_end, :] = packed4b_batch
            # 累加新增 keys 的统计（_count_cluster_keys 已支持 multi-batch，返回 [batch, kv_heads, B, K]）
            cluster_counts = self._count_cluster_keys(codebook_batch, codebook_batch.device)
            self.cluster_key_counts_gpu[ldx] += cluster_counts
            del codebook_batch, block_weight_batch, packed4b_batch, cluster_counts

            # 从 _sl_vnorm_cache 直接 copy 到 value_norm_gpu（norms 已在 decode 阶段增量维护）
            if self.enable_offload:
                self.value_norm_gpu[ldx][:, :, encode_start:encode_end] = self._sl_vnorm_cache[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens]
            else:
                self.value_norm_gpu[ldx][:, :, encode_start:encode_end] = self._sl_vnorm_cache[ldx][:, :, encode_start:encode_end]

            # ===========================================
            # 【Offload 模式】Update 时：
            # 1. 把 Local 最老的部分 offload 到 CPU（进入 retrieval zone）
            # 2. 滑动 GPU 上的 Local window（Local 新部分 + Update Buffer → 新 Local）
            if self.enable_offload and self.unified_keys_cpu is not None:
                # 1. 把 Local 最老的 num_new_tokens 个 tokens offload 到 CPU
                # 这些 tokens 从 GPU local 区域移出，进入 CPU retrieval zone
                # GPU 布局: [0:sink] [sink:sink+local] [sink+local:sink+local+update]
                # Local 最老部分: [sink : sink+num_new_tokens]
                
                cpu_write_start = self.cpu_valid_end[ldx]  # 第 ldx 层 CPU 缓存的有效结束位置+1，即本次 offload 的开始写入位置
                cpu_write_end = cpu_write_start + num_new_tokens  # 本次 offload 的结束写入位置
                
                # 【优化】使用预分配 buffer 替代 clone()，避免内存分配开销
                # 注意：num_new_tokens 应该等于 dynamic_update_interval
                src_keys = self._offload_keys_buffer[ldx][:, :, :num_new_tokens, :]
                src_values = self._offload_values_buffer[ldx][:, :, :num_new_tokens, :]
                src_keys.copy_(self.unified_keys_gpu[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens, :])
                src_values.copy_(self.unified_values_gpu[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens, :])
                # 保留引用：避免异步拷贝过程中源 tensor 被 allocator 复用
                self._inflight_offload_buffers[ldx] = (src_keys, src_values)
                self._layer_offload_inflight[ldx] = True
                
                # 在 copy stream 中异步拷贝到 CPU（与主流并行） 
                self._main_events[ldx].record()
                with torch.cuda.stream(self._copy_stream):
                    self._main_events[ldx].wait()
                    # 将 src_keys  拷贝到 dst_keys 
                    dst_keys = self.unified_keys_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                    dst_values = self.unified_values_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                    dst_keys.copy_(src_keys, non_blocking=True)
                    dst_values.copy_(src_values, non_blocking=True)
                    self._copy_events[ldx].record()
                    
                self.cpu_valid_end[ldx] = cpu_write_end
                
                # 2. 滑动 GPU Local window（可以与 CPU 拷贝并行）
                # 把 [sink+num_new_tokens : sink+local+num_new_tokens] 移动到 [sink : sink+local]
                # 即：原 Local 新部分 + 原 Update Buffer → 新 Local
                old_local_end = self.sink_size + self.local_size
                old_buffer_end = old_local_end + num_new_tokens
                new_local_start_src = self.sink_size + num_new_tokens  # old_local_start + num_new_tokens
                
                # 只有当 num_new_tokens < local_size 时源和目标才会重叠，需要 clone
                if num_new_tokens < self.local_size:
                    new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                    new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                else:
                    new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                    new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                self.unified_keys_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_keys
                self.unified_values_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_values
                if num_new_tokens < self.local_size:
                    new_local_vnorms = self._sl_vnorm_cache[ldx][:, :, new_local_start_src:old_buffer_end].clone()
                else:
                    new_local_vnorms = self._sl_vnorm_cache[ldx][:, :, new_local_start_src:old_buffer_end]
                self._sl_vnorm_cache[ldx][:, :, self.sink_size:old_local_end] = new_local_vnorms
                if num_new_tokens < self.local_size:
                    del new_local_keys, new_local_values, new_local_vnorms
                
                # if ldx == 0:
                #     print(colored(f"[Offload] GPU滑窗完成: 新Local=[{self.sink_size}:{old_local_end}] (来自[{new_local_start_src}:{old_buffer_end}]), CPU已存{cpu_write_end}个token", 'yellow'))
        
        # 等待所有层的 offload 完成，这样 retrieval 阶段不需要再检查
        if self.enable_offload:
            self._sync_offload_copies()
        
        self.steady_size = self.sink_size + self.local_size





    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        统一架构:Decode阶段直接append到unified_keys_gpu
        每生成1个token,直接追加到当前序列末尾
        当积累 dynamic_update_interval 个新tokens时,触发动态更新(编码新tokens到retrieval zone)
        """
        # nvtx.range_push(f"decode_update_layer_{layer_idx}")
        
        # ============ Step 1: 先写入当前 token ============
        # nvtx.range_push("kv_write")
        key_to_add = key_states[:, 0, :, :]
        value_to_add = value_states[:, 0, :, :]
        
        if self.enable_offload:
            # 【Offload 模式】GPU 布局: [Sink][Local][UpdateBuffer]
            current_len = self.current_seq_len[layer_idx]
            steady_threshold = self.steady_size  # 预计算：sink_size + local_size
            if current_len < steady_threshold:
                # Local 还没填满，继续追加到 Local 区域
                gpu_write_pos = current_len
            else:
                # Local 已满，写入 Update Buffer
                # 修复：统一用 current_len 计算 buffer_offset，
                # 避免短 prefill 下 _decode_step_counter 与实际 buffer 位置不对齐
                if self.input_length >= steady_threshold:
                    # 长 prefill：用 decode counter（counter 与 buffer 填充是 1:1 对应的）
                    tokens_in_buffer = self._decode_step_counter - self._buffer_base_counter
                else:
                    # 短 prefill：用序列长度溢出量（因为 counter 从 decode 开始计数，
                    # 但 buffer 从 current_len 超过 steady_threshold 才开始填充）
                    tokens_in_buffer = current_len - steady_threshold
                buffer_offset = tokens_in_buffer % self.dynamic_update_interval
                gpu_write_pos = steady_threshold + buffer_offset
            
            self.unified_keys_gpu[layer_idx][:, :, gpu_write_pos, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, gpu_write_pos, :] = value_to_add
            self._sl_vnorm_cache[layer_idx][:, :, gpu_write_pos] = torch.linalg.vector_norm(
                value_to_add, ord=2, dim=-1).to(torch.float32)

        else:
            # 【非 Offload 模式】按全局位置追加
            current_len = self.current_seq_len[layer_idx]
            self.unified_keys_gpu[layer_idx][:, :, current_len, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, current_len, :] = value_to_add
            self._sl_vnorm_cache[layer_idx][:, :, current_len] = torch.linalg.vector_norm(
                value_to_add, ord=2, dim=-1).to(torch.float32)

        # 更新全局序列长度
        self.current_seq_len[layer_idx] += 1
        # nvtx.range_pop()  # kv_write
        
        # ============ Step 2: 在第一层检查是否触发 update（写入后检查！）============
        if layer_idx == 0:
            # print(f"[AFTER WRITE] current_seq_len[0]={self.current_seq_len[0]} (compute 会用这个值)")
            # 递增计数器（在写入 token 之后）
            self._decode_step_counter += 1
            self.context += 1
            
            # 处理延迟的 buffer base 更新（base 应设为触发时的 counter 值，即当前 counter - 1）
            if self._pending_buffer_reset:
                self._buffer_base_counter = self._decode_step_counter - 1
                self._pending_buffer_reset = False
                # print(f"[BASE RESET] old_base={old_base} → new_base={self._buffer_base_counter} (counter={self._decode_step_counter})")
            
            # 检查是否需要触发 update（Buffer 刚好写满 dynamic_update_interval 个）
            current_total_len = self.current_seq_len[0]  # 现在包含刚写入的 token！
            steady_threshold = self.steady_size  # 预计算：sink_size + local_size
            
            if current_total_len > steady_threshold:
                if self.input_length >= steady_threshold:
                    # 长 prefill：用 decode counter 计算 buffer 内 token 数
                    buffer_tokens = self._decode_step_counter - self._buffer_base_counter
                else:
                    # 短 prefill：用序列长度溢出计算
                    buffer_tokens = current_total_len - steady_threshold
                
                # Buffer 刚好满（256 个）时触发
                if buffer_tokens > 0 and buffer_tokens % self.dynamic_update_interval == 0:
                    # print(f"[Decode] 触发 _update_polar_index, buffer_tokens={buffer_tokens}, step={self._decode_step_counter}")
                    # nvtx.range_push("update_polar_index")
                    self._update_polar_index()
                    self._pending_buffer_reset = True
                    # nvtx.range_pop()
        
        # nvtx.range_pop()  # decode_update_layer_{layer_idx}
        return None, None


    

    # ============================================================================
    # 4-bit 合并量化：1-bit 符号 + 3-bit 幅值 → 4-bit（每 byte 存 2 个值）
    def _pack_4bit_codes(self, signs: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """
        将 1-bit 符号 + 3-bit 幅值合并打包成 4-bit。
        Args:
            signs: [..., m] bool, True=正(>=0), False=负(<0)
            magnitudes: [..., m] uint8, 0-7 (3-bit 幅值码)
        Returns:
            packed: [..., m//2] uint8, 每个 byte 存 2 个 4-bit 值 (m=8 时为 4 bytes)
        """
        # 4-bit = (sign << 3) | magnitude, 范围 0-15
        v4bit = (signs.to(torch.uint8) << 3) | magnitudes  # [..., m], uint8
        # 每 2 个 4-bit 打包成 1 个 byte: (v_odd << 4) | v_even
        v_even = v4bit[..., 0::2]  # 偶数索引: v0, v2, v4, v6
        v_odd = v4bit[..., 1::2]   # 奇数索引: v1, v3, v5, v7
        packed = (v_odd << 4) | v_even  # [..., m//2]
        return packed.to(torch.uint8).contiguous()



    def _unpack_4bit_codes(self, packed: torch.Tensor, m: int = 8) -> tuple:
        """
        从 4-bit packed 解出符号和幅值。
        Args:
            packed: [..., m//2] uint8 (m=8 时为 4 bytes)
        Returns:
            s_sign: [..., m] int8, ±1
            t_codes: [..., m] int64, 0-7 (用于查表 _mag_centers)
        """
        # 解包 2 个 4-bit 值
        v_even = packed & 0x0F           # [..., m//2], 低 4 位
        v_odd = (packed >> 4) & 0x0F     # [..., m//2], 高 4 位
        # 交错合并: [v0, v1, v2, v3, ...] from [even0, even1, ...] and [odd0, odd1, ...]
        v4bit = torch.stack([v_even, v_odd], dim=-1).view(*packed.shape[:-1], m)  # [..., m]
        # 拆分符号和幅值
        sign_bits = (v4bit >> 3) & 1     # [..., m], 0/1
        s_sign = (2 * sign_bits - 1).to(torch.int8)  # 0->-1, 1->+1
        t_codes = (v4bit & 0x7).to(torch.int64)      # [..., m], 0-7
        return s_sign, t_codes



    def _encode_query_blocks(self, query: torch.Tensor) -> tuple:
        """
        将 query 编码为 SRHT 后的 blocks（用于 3-bit rerank）。
        Args:
            query: [bs, kv_heads, head_dim] 原始 query（未归一化）
        Returns:
            q_blocks: [bs, kv_heads, B, m] (来自单位化 query 的 SRHT 结果，包含 block 半径)
            q_norm: [bs, kv_heads, 1] query 的模长
        """
        q_norm = torch.linalg.vector_norm(query, ord=2, dim=-1, keepdim=True)  # [bs, kv, 1]
        query_normalized = query / q_norm.clamp(min=1e-8)
        Y = query_normalized
        for round_idx in range(DEFAULT_SRHT_ROUNDS):
            Y = Y * self.quantizer.srht_diagonal_signs[round_idx]
            Y = hadamard_transform(Y, scale=self._hadamard_scale)
        q_blocks = Y.view(query.shape[0], query.shape[1], self.polar_B, self.polar_m).contiguous()
        return q_blocks, q_norm




    # 批量版本的编码和索引构建函数
    def batch_encode(self, Y, layer_idx=None, return_rabitq: bool = True):
        """
        4-bit RaBitQ 编码：1-bit 符号 + 3-bit 幅值
        Args:
            Y: [bs, kv_heads, seq_len, head_dim] GPU tensor
            layer_idx: 层索引(可选,用于调试)
            return_rabitq: 是否返回完整编码（默认 True，False 时跳过 4-bit 量化）
        Returns:
            codebook: [bs, kv_heads, B, seq_len] uint8, bitpacked 符号（已转置，可直接存储）
            block_weight: [bs, kv_heads, B, seq_len] bf16, 已融合 (r/α) * ||k||（return_rabitq=False 时为 None）
            packed_4bit: [bs, kv_heads, B, seq_len, m//2] uint8, 4-bit packed（return_rabitq=False 时为 None）
        """
        bs, kv_heads, seq_len, head_dim = Y.shape
        key_norm = torch.linalg.vector_norm(Y, ord=2, dim=-1, keepdim=True)  # [bs, kv_heads, seq_len, 1]
        Y_normalized = Y / key_norm.clamp(min=1e-8)  # [bs, kv_heads, seq_len, head_dim]
        key_norm = key_norm.squeeze(-1)  # [bs, kv_heads, seq_len]
        
             
        # 步骤2:Layer-specific SRHT旋转(直接在4D tensor上操作)
        Y_rotated = Y_normalized
        for round_idx in range(DEFAULT_SRHT_ROUNDS):
            Y_rotated = Y_rotated * self.quantizer.srht_diagonal_signs[round_idx]
            Y_rotated = hadamard_transform(Y_rotated, scale=self._hadamard_scale)
        
                
        # 步骤3:分块 [bs, kv_heads, seq_len, head_dim] -> [bs, kv_heads, seq_len, B, m]
        Y_blocks = Y_rotated.view(bs, kv_heads, seq_len, self.quantizer.B, self.quantizer.m)
        
        # 步骤4: 方向量化 - 直接 bitpack sign(u)
        # 码本就是全体 256 个 sign pattern，直接 bitpack 比 256-way dot + argmax 更快、更数值稳定
        bits = (Y_blocks >= 0)  # [bs, kv, seq, B, m], bool
        angular_codes_u8 = (bits * self._bitpack_shifts).sum(dim=-1, dtype=torch.uint8)  # [bs, kv, seq, B]
        codebook = angular_codes_u8.permute(0, 1, 3, 2).contiguous()  # [bs, kv, B, seq]
        

        # 如果不需要 RaBitQ rerank，只返回 codebook（用于碰撞检索）
        if not return_rabitq:
            return codebook, None, None
        
        # ===== 以下是 4-bit RaBitQ rerank 需要的计算 =====
        # 计算每个 block 的半径 r_{k,b}
        block_radii = torch.linalg.vector_norm(Y_blocks, ord=2, dim=-1)  # [bs, kv_heads, seq_len, B]
        # 方向归一化
        denom = block_radii.unsqueeze(-1).clamp(min=1e-8)
        directions = Y_blocks / denom  # [bs, kv_heads, seq_len, B, m]
        del Y_blocks, denom  # 释放不再需要的 tensor

        # ===== 4-bit 合并量化：1-bit 符号 + 3-bit 幅值 =====
        # t_j = bucketize(|u_j|, τ) ∈ {0,1,2,3,4,5,6,7}
        abs_u = directions.abs()  # 单位向量的大小，[bs, kv, seq, B, m] bf16
        del directions  # 释放 directions，只保留 abs_u
        t_codes = torch.bucketize(abs_u, self._mag_thresholds)  # 返回 int64
        
        # ===== 计算 alpha^(4b) = Σ a[t_j] * |u_j| =====
        # 先用 t_codes (int64) 做索引，再转 uint8，避免同时持有两个大 tensor
        a_vals = self._mag_centers[t_codes]  # [bs, kv, seq, B, m] bf16
        t_codes_u8 = t_codes.to(torch.uint8)
        del t_codes  # 释放 int64 tensor 以节省显存
        # 4-bit pack: 1-bit sign + 3-bit magnitude → 4 bytes per block (m=8)
        # bits: [bs, kv, seq, B, m] bool, t_codes_u8: [bs, kv, seq, B, m] uint8
        packed_4bit = self._pack_4bit_codes(bits, t_codes_u8)  # [bs, kv, seq, B, m//2]
        alpha4b = (a_vals * abs_u).sum(dim=-1, dtype=torch.float32)  # [bs, kv, seq, B]，abs_u 已是 bf16
        alpha4b = alpha4b.clamp(min=1e-6)
        # weight = (r / α) * ||k||（RaBitQ 公式，必须除以 alpha）
        block_weight = (block_radii / alpha4b) * key_norm.unsqueeze(-1)  # [bs, kv, seq, B]
        block_weight = block_weight.permute(0, 1, 3, 2).to(torch.bfloat16).contiguous()  # [bs, kv, B, seq]
        packed_4bit = packed_4bit.permute(0, 1, 3, 2, 4).contiguous()  # [bs, kv, B, seq, m//2]
        return codebook, block_weight, packed_4bit
    



    def _count_cluster_keys(self, codebook, device):
        """
        统计每个 cluster 包含的 key 数量（支持 multi-batch）
        Args:
            codebook: [batch, kv_heads, B, kv_len]
            device: 目标设备
        Returns:
            counts: [batch, kv_heads, B, K_omega] 每个 cluster 的 key 数量
        """
        batch_size, kv_heads, B, kv_len = codebook.shape
        actual_K = self.polar_K_omega
        num_groups = kv_heads * B
        
        # 逐 batch 处理（bincount 不支持 batch 维度）
        counts_list = []
        for b in range(batch_size):
            codebook_b = codebook[b]  # [kv_heads, B, kv_len]
            codebook_flat = codebook_b.reshape(-1).long()
            group_ids = torch.arange(num_groups, device=device).view(kv_heads, B, 1).expand(-1, -1, kv_len).reshape(-1)
            global_ids = codebook_flat + group_ids * actual_K
            counts_flat = torch.bincount(global_ids, minlength=num_groups * actual_K).to(torch.int32)
            counts_list.append(counts_flat.view(kv_heads, B, actual_K))
        
        return torch.stack(counts_list, dim=0)  # [batch, kv_heads, B, K_omega]
    
    
    
    # 单层Decode 阶段的 Attention 计算
    # 输入:query 向量,  输出:attention 结果
    def compute(self, queries, layer_idx):
        """
        输入: queries [batch_size, seq_len, num_heads, head_dim]  seq_len=1
        输出: attn_out [batch_size, seq_len, num_heads, head_dim] seq_len=1
        Decode计算:
        - 当 total_length < 1000 时:全量 attention(不检索)
        - 当 total_length >= 1000 时:使用Polar ANN检索top-k tokens
        """
        # nvtx.range_push(f"decode_layer_{layer_idx}")
        # nvtx.range_push("decode_prepare_query")
        
        # 统一架构：按需计算区域索引，避免每步 decode 都构造 index tensor
        device = self.device
        total_length = self.current_seq_len[layer_idx]
        local_start = self._compute_local_start(layer_idx)
        local_len = total_length - local_start if local_start < total_length else 0
        retrieval_start = self.sink_size
        retrieval_length = max(0, local_start - self.sink_size)
        
        # 计算有效检索长度（避免访问尚未编码/offload 的区域）
        encoded_len = self._actual_retrieval_length or 0
        cpu_len = self.cpu_valid_end[layer_idx] if (self.enable_offload and self.cpu_valid_end is not None) else 0
        if self.enable_offload and self.cpu_valid_end is not None:
            # Offload 模式：cpu_len 是实际限制（cpu_len <= encoded_len <= retrieval_len）
            valid_length = cpu_len
        else:
            # 非 Offload 模式：需要同时考虑理论值和实际编码值
            valid_length = min(retrieval_length, encoded_len)
        retrieval_end_effective = retrieval_start + valid_length
        
        # 准备 query: [bs, 1, num_heads, head_dim] -> per-kv-head 平均 query [bs, kv_heads, 1, head_dim]
        bs, seqlen_q, head_q, head_dim = queries.shape
        query_group = queries[:, 0, :, :]  # [bs, num_heads, head_dim]
        queries_reshaped = query_group.view(bs, self.kv_head, self.group_size, self.head_dim)
        query_avg_gpu = queries_reshaped.mean(dim=2)   # [bs, kv_heads, head_dim]
        query_batch = query_avg_gpu.unsqueeze(2)       # [bs, kv_heads, 1, head_dim]
        # nvtx.range_pop()  # decode_prepare_query
        
        # 判断是否需要检索
        # total_length 至少要 > sink + local + final_topk*2，否则 retrieval zone 太小
        min_total_for_retrieval = max(self.FULL_ATTENTION_THRESHOLD, self.sink_size + self.local_size + self.final_topk * 2)
        use_full_attention = (total_length < min_total_for_retrieval) or (valid_length <= 0)


        if use_full_attention:
            # ============ 全量 Attention 模式 ============
            sink_indices = torch.arange(0, self.sink_size, device=device, dtype=torch.long)
            if local_len > 0:
                local_indices = torch.arange(local_start, total_length, device=device, dtype=torch.long)
            else:
                local_indices = torch.zeros(0, device=device, dtype=torch.long)
            concat_keys, concat_values = self._get_full_kv_for_attention(layer_idx, sink_indices, local_indices, retrieval_start, retrieval_length, device, queries.dtype)
            if layer_idx == 0 and not hasattr(self, '_full_attn_printed'):
                print(f"[Full Attention] threshold={min_total_for_retrieval} (FULL_ATTN={self.FULL_ATTENTION_THRESHOLD}, sink={self.sink_size}, local={self.local_size}, topk={self.final_topk})", flush=True)
                print(f"[Full Attention] enable_offload={self.enable_offload}, kv_shape={concat_keys.shape}, total_len={total_length}, retrieval_len={retrieval_length}, valid_len={valid_length}", flush=True)
                self._full_attn_printed = True
            if layer_idx == 0 and total_length % 100 == 0:
                print(f"[Mode] total_len={total_length}/{min_total_for_retrieval}, mode=FULL_ATTN, kv_len={concat_keys.shape[2]}, offload={self.enable_offload}", flush=True)
           
           
            # 全量 Flash Attention (GQA handled internally by flash_attn)
            k_cache = concat_keys.transpose(1, 2)   # [bs, seq_len, kv_heads, head_dim]
            v_cache = concat_values.transpose(1, 2)
            attn_out = flash_attn_with_kvcache_compat(
                q=queries,    # [bs, 1, num_heads, head_dim]
                k_cache=k_cache,
                v_cache=v_cache,
                causal=False
            )
            del concat_keys, concat_values, k_cache, v_cache
        else:
            if layer_idx == 0 and not hasattr(self, '_retrieval_printed'):
                print(f"[Retrieval Mode] total_len={total_length} >= {min_total_for_retrieval}, valid_len={valid_length}, retrieval_len={retrieval_length}", flush=True)
                self._retrieval_printed = True
            if layer_idx == 0 and total_length % 100 == 0:
                print(f"[Mode] step total_len={total_length}, threshold={min_total_for_retrieval}, mode=RETRIEVAL, valid_len={valid_length}", flush=True)
            # ============ 检索模式 (Per-Query-Head) ============
            # nvtx.range_push("get_retrieval_data")
                
            # 按绝对位置获取 codebook、weight 和 4-bit packed (kv_head 级别)
            codebook_gpu = self.codebook_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective]
            if USE_RABITQ_RERANK:
                key_block_weight_gpu = self.key_block_weight_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective]
                key_4bit_gpu = self.key_4bit_packed_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective, :]
            else:
                key_block_weight_gpu = None
                key_4bit_gpu = None

            cluster_key_counts_expanded = self.cluster_key_counts_gpu[layer_idx]  # [bs, kv_heads, B, num_clusters]

            # 【Offload 模式】Retrieval 起点固定为 sink_size
            assert not self.enable_offload or retrieval_start == self.sink_size, f"Offload 模式下 retrieval_start({retrieval_start}) 必须等于 sink_size({self.sink_size})"
            # nvtx.range_pop()  # get_retrieval_data

            # 自适应调节 candidate_ratio 和 collision_ratio
            retrieval_len = valid_length  # retrieval zone 的实际长度
            adaptive_candidate_ratio, adaptive_collision_ratio = self._get_adaptive_ratios(retrieval_len)
            # 批量调用 collision_based_topk_batch (per-query-head: [bs, num_heads, ...])
            # nvtx.range_push("collision_based_topk_batch")
            # 始终走 adaptive-K 路径：返回全部候选的 raw scores/indices，下游每步按 benefit 重排选 k
            # 前两个返回值在 adaptive 路径下为 None（省掉一次按 approx_scores 的冗余全排序）
            _, _, all_cand_scores, all_cand_indices = self.collision_based_topk_batch(
                query=query_batch,
                codebook=codebook_gpu,
                key_block_weight=key_block_weight_gpu,
                key_4bit=key_4bit_gpu,
                collision_ratio=adaptive_collision_ratio,
                candidate_ratio=adaptive_candidate_ratio,
                layer_idx=layer_idx,
                cluster_key_counts=cluster_key_counts_expanded,
                adaptive_topk=True,
            )
            # nvtx.range_pop()

            # ============ Sparse Attention: Sink+Local + Top-k 一体化 ============
            # 注意：Offload 与非 Offload 的 unified_*_gpu 物理布局不同
            # - Offload:   [Sink][Local][UpdateBuffer]（Local 紧跟 Sink，可直接前缀切片）
            # - NonOffload:[0..t] 时间顺序（Local 在序列末尾，不能用前缀切片）
            if self.enable_offload and self.unified_values_cpu is not None:
                sink_local_end = self.sink_size + local_len
                sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
                sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]
            else:
                sink_idx = torch.arange(0, self.sink_size, device=device, dtype=torch.long)
                if local_len > 0:
                    local_idx = torch.arange(local_start, total_length, device=device, dtype=torch.long)
                    all_idx = torch.cat([sink_idx, local_idx], dim=0)
                else:
                    all_idx = sink_idx
                sink_local_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_idx)
                sink_local_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_idx)

            # 从增量缓存获取 sink+local value norms（避免每步重算 vector_norm）
            if self.enable_offload and self.unified_values_cpu is not None:
                sl_vnorm = self._sl_vnorm_cache[layer_idx][:, :, :sink_local_end]
            else:
                sl_vnorm = torch.index_select(self._sl_vnorm_cache[layer_idx], 2, all_idx)

            
            
            # ============ Adaptive Top-k Selection (value-weighted) ============
            # 每一步都重算 adaptive_k：candidate 集合在两次 buffer flush 之间不变，
            # 但 query 每步不同，最优 k 也每步不同。把缓存拿掉换来更紧的 k。
            adaptive_k, topk_indices_batch, global_k = self._compute_adaptive_topk(
                query_avg=query_avg_gpu,
                sink_local_keys=sink_local_keys,
                sl_vnorm=sl_vnorm,
                all_cand_scores=all_cand_scores,
                all_cand_indices=all_cand_indices,
                layer_idx=layer_idx,
                threshold=self.adaptive_topk_threshold,
            )

            # 每一步都收集每层 max k；最后一层算完打整行（一行 = 全部层的 max k）。
            # 复用 global_k（已经是 int），避免再触发一次 adaptive_k.max().item() 的 CPU sync。
            self._adaptive_k_log_max[layer_idx] = global_k
            
            if layer_idx == self.layer_num - 1:
                layer_max_list = [(-1 if v is None else v) for v in self._adaptive_k_log_max]
                row_str = " ".join(f"{v:>5d}" for v in layer_max_list)
                print(f"[AdaptiveK-VW] step={total_length} per_layer_max_k(L0..L{self.layer_num-1}): {row_str}", flush=True)
                self._adaptive_k_log_max = [None] * self.layer_num
            del all_cand_scores, all_cand_indices

            # 直接拼接 Sink/Local + Top-k，一次 Flash Attention
            is_offload = (self.enable_offload and self.unified_keys_cpu is not None and self.unified_values_cpu is not None)
            
            if is_offload:
                # 【UVA 路径】GPU indices + pinned CPU src → GPU dst buffer，一次 kernel 搞定
                # _compute_adaptive_topk 已保证 topk_indices_batch 是 long 且 contiguous
                self._h2d_gather_kv(
                    self.unified_keys_cpu[layer_idx],    # src keys: CPU pinned bf16
                    self.unified_values_cpu[layer_idx],  # src values: CPU pinned bf16
                    topk_indices_batch,                   # indices: GPU long contiguous
                    self._topk_keys_buffer,               # dst keys: GPU bf16, cap=adaptive_topk_max_k
                    self._topk_values_buffer,             # dst values: GPU bf16
                )
                # kernel 只写前 global_k 行，concat 时切片到有效区域
                topk_keys = self._topk_keys_buffer[:, :, :global_k, :]
                topk_values = self._topk_values_buffer[:, :, :global_k, :]
            else:
                keys_tensor = self.unified_keys_gpu[layer_idx]
                values_tensor = self.unified_values_gpu[layer_idx]
                topk_indices_src = topk_indices_batch + retrieval_start
                idx_exp = topk_indices_src.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                topk_keys = torch.gather(keys_tensor, 2, idx_exp)      # [bs, kv_heads, topk, d]
                topk_values = torch.gather(values_tensor, 2, idx_exp)  # [bs, kv_heads, topk, d]

            concat_keys = torch.cat([sink_local_keys, topk_keys], dim=2)
            concat_values = torch.cat([sink_local_values, topk_values], dim=2)
            k_cache = concat_keys.transpose(1, 2)   # [bs, seq_len, kv_heads, head_dim]
            v_cache = concat_values.transpose(1, 2)

            attn_out = flash_attn_with_kvcache_compat(
                q=queries,
                k_cache=k_cache,
                v_cache=v_cache,
                causal=False
            )
            del sink_local_keys, sink_local_values
            del topk_keys, topk_values
            del concat_keys, concat_values, k_cache, v_cache
        
        # nvtx.range_pop()  # decode_layer_{layer_idx}
        return attn_out




    # ============================================================================
    # Adaptive Top-k: 根据 attention mass 阈值自适应确定每个 head 的 k
    # ============================================================================
    def _compute_adaptive_topk(
        self,
        query_avg: torch.Tensor,
        sink_local_keys: torch.Tensor,
        sl_vnorm: torch.Tensor,
        all_cand_scores: torch.Tensor,
        all_cand_indices: torch.Tensor,
        layer_idx: int,
        threshold: float = 0.9,
    ):
        """
        基于 softmax(qk/sqrt(d)) * ||v||_2 自适应确定每个 kv head 的 retrieval k，
        并直接产出可喂给 sparse attention / h2d_gather_kv 的 topk_indices。

        Args:
            query_avg:        [bs, kv_heads, head_dim]
            sink_local_keys:  [bs, kv_heads, sl_len, head_dim]  预缓存的 (sink+local+update_buffer) 的 key
            sl_vnorm:         [bs, kv_heads, sl_len] 预缓存的 (sink+local+update_buffer) value L2 norms
            all_cand_scores:  [bs, kv_heads, cand_len] retrieval zone 候选的 RaBitQ 近似 QK 内积（未除 sqrt(d)）
            all_cand_indices: [bs, kv_heads, cand_len] 候选在 retrieval zone 中的相对索引
            layer_idx:        层索引（用于索引 value_norm_gpu）
            threshold:        累积 benefit 占比阈值
        Returns:
            adaptive_k:    [bs, kv_heads]            per-head 真实需要的 k
            topk_indices:  [bs, kv_heads, global_k]  long & contiguous，retrieval zone 相对索引，可直接喂 h2d_gather_kv
            global_k:      int                       max(adaptive_k)，dense tensor 第三维
        """
        cand_len = all_cand_scores.shape[-1]
        sl_len = sink_local_keys.shape[2]

        # 1. QK logits（在原始 candidate 顺序上计算；softmax 对顺序不敏感）
        logits_sl = torch.einsum('bhd,bhkd->bhk', query_avg, sink_local_keys).to(torch.float32) * self._attn_scale
        logits_cand = all_cand_scores.to(torch.float32) * self._attn_scale

        # 2. joint exp（替代 softmax）
        # 等价改写：判定式 sl_b + Σ_topk b_i >= τ·total_b 两边的 1/Z 是公因子，可约掉。因此无需做 softmax，直接用未归一化质量 m_i = exp(l_i - m) * v_i 即可。
        # 减全局 max logit 仅为 fp32 数值稳定，对判定/排序无影响（同样是公因子）。
        joint_logits = torch.cat([logits_sl, logits_cand], dim=-1)  # [bs, kv_heads, sl+cand]
        joint_logits_max = joint_logits.amax(dim=-1, keepdim=True)
        joint_exp = torch.exp(joint_logits - joint_logits_max)  # [bs, kv_heads, sl+cand]

        # 3. value norms (sl_vnorm 已从 _sl_vnorm_cache 增量维护，无需重算)
        # all_cand_indices 是 0-based（相对于 sliced retrieval zone），但 value_norm_gpu 按绝对位置存储（从 sink_size 开始），需要加偏移
        abs_cand_indices = all_cand_indices.long() + self.sink_size
        cand_vnorm = torch.gather(self.value_norm_gpu[layer_idx], dim=-1, index=abs_cand_indices)  # [bs, kv_heads, cand_len]
        joint_vnorm = torch.cat([sl_vnorm, cand_vnorm], dim=-1)  # [bs, kv_heads, sl+cand]

        # 4. benefit (未归一化质量) = exp(l - m) * value_norm
        benefit = joint_exp * joint_vnorm

        # 5. 按 benefit 降序选取 top-cap 候选（cap = adaptive_topk_max_k）。
        # 直接把排序范围截到 cap 个，长上下文场景下 sort O(N log N) → topk O(N log K)；
        sl_benefit = benefit[:, :, :sl_len].sum(dim=-1, keepdim=True)  # [bs, kv_heads, 1]
        cand_benefit = benefit[:, :, sl_len:]                           # [bs, kv_heads, cand_len] (raw candidate order)
        topk_cap = min(self.adaptive_topk_max_k, cand_len)
        cand_benefit_topk, sort_perm = torch.topk(cand_benefit, k=topk_cap, dim=-1, largest=True, sorted=True)
        cumsum_benefit = torch.cumsum(cand_benefit_topk, dim=-1)
        # total benefit = sl + 全部 cand；用 cand_benefit.sum 而不是 cumsum_benefit[..., -1:]
        # 是因为 topk 之外的尾部也要参与分母（否则 threshold 永远在 cap 处达成）
        total_benefit_all = sl_benefit + cand_benefit.sum(dim=-1, keepdim=True)
        reached = (sl_benefit + cumsum_benefit >= threshold * total_benefit_all)
        adaptive_k = reached.long().argmax(dim=-1) + 1  # [bs, kv_heads]
        never_reached = ~reached.any(dim=-1)
        adaptive_k[never_reached] = topk_cap  # 没达成阈值就取到 cap（等价于原来取 cand_len 后再 clamp 到 cap）

        # total benefit ≤ 0 时回退到 1（避免 0 候选）
        bad_total = (total_benefit_all.squeeze(-1) <= 0)
        if bad_total.any():
            adaptive_k = torch.where(bad_total, torch.ones_like(adaptive_k), adaptive_k)

        # 6. 切到 max K 并 gather 出物理位置（retrieval zone 相对索引）。
        # global_k = max(adaptive_k)：per-head K 不同但 dense tensor 必须等长，
        # 这一维的 padding 是 GQA 多 head 不同稀疏度的固有代价。
        # .item() 触发一次 CPU sync；调用方需要的话可直接复用 global_k，避免再算一遍 adaptive_k.max().item()。
        # torch.gather 接受 non-contiguous index（按 stride 计算偏移），且输出永远 contiguous，可直接喂 h2d_gather_kv。
        global_k = int(adaptive_k.max().item())
        sort_perm_topk = sort_perm[:, :, :global_k]
        topk_indices = torch.gather(all_cand_indices, dim=2, index=sort_perm_topk)

        return adaptive_k, topk_indices, global_k










    
    # ============================================================================
    # 批量并行的 collision_based_topk 函数    
    def collision_based_topk_batch(self, query, codebook, key_block_weight=None,
                                     key_4bit=None, collision_ratio=0.2, 
                                     candidate_ratio=0.2, layer_idx=None,
                                     cluster_key_counts=None,
                                     adaptive_topk=False):
        """
        两阶段碰撞 Top-k 检索（碰撞粗排 → RaBitQ 精排）
        
        Stage 1: 碰撞计数 → coarse filter 取 candidate_ratio * kv_len 个候选
        Stage 2: RaBitQ 4-bit 近似内积 → 精排取 final_topk 个
        
        Args:
            query: [bs, heads, 1, head_dim]（原始 query,未归一化;heads 可以是 kv_heads 或 num_heads)
            codebook: [bs, heads, B, kv_len]
            key_block_weight: [bs, heads, B, kv_len] bf16, 已融合 (r/alpha) * ||k||
            key_4bit: [bs, heads, B, kv_len, 4] 4-bit packed (1-bit sign + 3-bit magnitude)
            collision_ratio: 碰撞比例（控制每个子空间激活多少 clusters)
            candidate_ratio: 候选集比例(Stage 1 输出 = candidate_ratio * kv_len)
            cluster_key_counts: [bs, heads, B, num_clusters] 预计算的 cluster key 计数（可选）
            adaptive_topk: bool, 若 True 则额外返回全部候选的 approx_scores 和 coarse_indices
        Returns:
            topk_indices: [bs, heads, final_topk] 精排后的 key 索引
            topk_scores: [bs, heads, final_topk] 对应的 RaBitQ 近似内积 (未乘 scale)
            (若 adaptive_topk=True，额外返回:)
            all_cand_scores: [bs, heads, candidate_len] 全部候选的近似内积
            all_cand_indices: [bs, heads, candidate_len] 全部候选的索引
        Note:
            返回 indices + scores,调用方用 scores 做 attention logits(乘 scale 后 softmax)
            key_block_weight 已融合 key_norm,省去一次 gather
        """
        # nvtx.range_push("_encode_query_blocks")
        bs, kv_heads, qlen, head_dim = query.shape
        assert qlen == 1, "Decode阶段每次只处理1个token"        
        query_squeezed = query.squeeze(2)  # [bs, kv_heads, head_dim]=[1, 8, 128]
        # 3-bit rerank: 预计算 q_blocks 和 q_norm（只做一次 SRHT）
        q_blocks, q_norm = self._encode_query_blocks(query_squeezed)  # [bs,kv,B,m], [bs,kv,1]
        # nvtx.range_pop()
        
        # Step 2+3 融合: 计算碰撞计数 (get_topk_clusters + update_cache_cnt)
        # nvtx.range_push("collision_fused")
        cache_cnt = self.get_cache_cnt_fused(q_blocks, codebook, collision_ratio, layer_idx, cluster_key_counts=cluster_key_counts)
        # nvtx.range_pop()
        
        # Step 4: 4-bit RaBitQ rerank（key_block_weight 已融合 key_norm）
        if not (USE_RABITQ_RERANK and key_block_weight is not None and key_4bit is not None):
            raise ValueError(
                "本模块仅支持 USE_RABITQ_RERANK=True 且 key_block_weight/key_4bit 必须提供 "
                "(exact rerank 路径已删除)"
            )
        # nvtx.range_push("get_candidate_cache_with_rabitq_rerank")
        rerank_result = self.get_candidate_cache_with_rabitq_rerank(
            cache_cnt=cache_cnt,
            candidate_ratio=candidate_ratio,
            q_blocks=q_blocks,
            key_4bit=key_4bit,
            key_block_weight=key_block_weight,
            q_norm=q_norm,
            layer_idx=layer_idx,
            return_all_candidates=adaptive_topk,
        )
        # nvtx.range_pop()
        if adaptive_topk:
            topk_indices, topk_scores, all_cand_scores, all_cand_indices = rerank_result
        else:
            topk_indices, topk_scores = rerank_result
        if adaptive_topk:
            return topk_indices, topk_scores, all_cand_scores, all_cand_indices
        return topk_indices, topk_scores
    
    



    def get_candidate_cache_with_rabitq_rerank(
        self, cache_cnt, candidate_ratio, q_blocks, key_4bit, key_block_weight,
        q_norm, layer_idx=None, return_all_candidates=False,
    ):
        """
        两阶段检索：碰撞粗排 → 4-bit RaBitQ 精排
        Stage 1 (Coarse): cache_cnt [bs, heads, kv_len] → coarse_indices [bs, heads, candidate_len]
                          candidate_len = ceil(candidate_ratio * kv_len)
        Stage 2 (Rerank): RaBitQ 近似内积对 candidate_len 个候选打分 → 取 top final_topk
        Args:
            cache_cnt: [bs, heads, kv_len] 碰撞计数
            candidate_ratio: 候选集比例 (Stage 1 输出 = candidate_ratio * kv_len)
            q_blocks: [bs, heads, B, m] query 的 SRHT 分块结果
            key_4bit: [bs, heads, B, kv_len, 4] 4-bit packed (1-bit sign + 3-bit magnitude)
            key_block_weight: [bs, heads, B, kv_len] bf16, 已融合 (r/α) * ||k||
            q_norm: [bs, heads, 1] query 的模长 ||q||
            return_all_candidates: bool, 若 True 则额外返回全部候选的 approx_scores 和 coarse_indices
        Returns:
            final_indices: [bs, heads, final_topk] 精排后的 key 索引
            final_scores: [bs, heads, final_topk] 对应的 RaBitQ 近似内积 (未乘 scale)
            (若 return_all_candidates=True，额外返回:)
            approx_scores: [bs, heads, candidate_len] 全部候选的近似内积
            coarse_indices: [bs, heads, candidate_len] 全部候选的索引
        """        
        # ===== 4-bit RaBitQ rerank (融合 CUDA kernel) =====
        # 0. Coarse filter and truncate
        # torch.cuda.synchronize()  # 取消注释此行来测试
        # nvtx.range_push("rerank_0_coarse_filter")
        coarse_indices, max_candidates = self._coarse_filter_truncate(cache_cnt, candidate_ratio, layer_idx)
        # nvtx.range_pop()
            
        # 1-3. 融合 CUDA kernel: Gather + Unpack + Score
        # 目前 CUDA kernel 只支持 B=16, m=8，其他情况回退到 PyTorch 实现
        if self.polar_B == 16 and self.polar_m == 8:
            # nvtx.range_push("rerank_1_3_fused_cuda")
            approx_scores, v4b_all = partial_rerank_cuda(
                coarse_indices.contiguous(),
                key_4bit,
                key_block_weight,
                q_blocks.contiguous(),
                q_norm.contiguous(),
                self._mag_centers.contiguous(),
                self.polar_B,
                self.polar_m,
                kernel=self._rerank_kernel  # 使用预加载的 kernel
            )
            # nvtx.range_pop()
            
        else:
            # PyTorch 回退实现
            # nvtx.range_push("rerank_1_gather")
            idx_code = coarse_indices.unsqueeze(2).expand(-1, -1, self.polar_B, -1)
            cand_weight = torch.gather(key_block_weight, dim=3, index=idx_code)
            idx_4bit = coarse_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, self.polar_B, -1, key_4bit.shape[-1])
            cand_4bit = torch.gather(key_4bit, dim=3, index=idx_4bit)
            # nvtx.range_pop()
            
            # nvtx.range_push("rerank_2_unpack_reconstruct")
            orig_shape = cand_4bit.shape[:-1]
            s_sign, t_codes = self._unpack_4bit_codes(cand_4bit.view(-1, cand_4bit.shape[-1]), m=self.polar_m)
            s_sign = s_sign.view(*orig_shape, self.polar_m)
            t_codes = t_codes.view(*orig_shape, self.polar_m)
            a_vals = self._mag_centers[t_codes]
            # nvtx.range_pop()
            
            # nvtx.range_push("rerank_3_score")
            v4b = a_vals * s_sign.to(a_vals.dtype)
            q_exp = q_blocks.unsqueeze(3)
            per_block_scores = (q_exp * v4b).sum(dim=-1)
            cosine_proxy = (per_block_scores * cand_weight).sum(dim=2)
            approx_scores = cosine_proxy * q_norm
            # nvtx.range_pop()
               
        # 4. Rerank (topk + gather)
        # nvtx.range_push("rerank_4_topk_gather")
        # adaptive 模式：下游 _compute_adaptive_topk 会按 benefit
        # (m_i = exp(l_i)·||v_i||) 重新排序并选 k，这里按 approx_scores 的全排序
        if return_all_candidates:
            return None, None, approx_scores, coarse_indices

        # 非 adaptive 模式：按 approx_scores 排序得到 final top-K（保持向后兼容，
        # 该模块当前实际不会走这条路径，但接口保留）
        actual_topk = approx_scores.shape[-1]
        _, rerank_indices = torch.topk(approx_scores, k=actual_topk, dim=-1)
        final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)
        final_scores = torch.gather(approx_scores, dim=2, index=rerank_indices)
        # nvtx.range_pop()
        return final_indices, final_scores

   

    
    def get_cache_cnt_fused(self, q_blocks, codebook, collision_ratio, layer_idx, cluster_key_counts=None):
        """
        融合版本: 将 get_topk_clusters + update_cache_cnt 合并为单次 CUDA 调用
        流程:
            1. 计算 query-cluster scores 并排序
            2. 获取预计算的 cluster_key_counts
            3. 计算 tier_counts 阈值
            4. 调用融合的 CUDA kernel (内部做 gather + cumsum + collision counting)
        Args:
            q_blocks: [bs, heads, B, m] query 在各子空间的编码 (heads 可以是 kv_heads 或 num_heads)
            codebook: [bs, heads, B, kv_len] 每个 key 的 cluster ID (uint8)
            collision_ratio: 碰撞检测范围比例
            layer_idx: 当前层索引
            cluster_key_counts: [bs, heads, B, num_clusters] 预计算的 cluster key 计数（可选，None 时从 self 获取）
        Returns:
            cache_cnt: [bs, heads, kv_len] 碰撞计数
        """
        kv_len = codebook.shape[3]
        # Step 1: 计算 scores 并排序得到 sorted_cluster_ids
        scores = torch.matmul(q_blocks, self._V_omega_gpu_T)  # [bs, heads, B, K_omega]
        sorted_cluster_ids = scores.argsort(dim=-1, descending=True).to(torch.int64)
        # Step 2: 获取预计算的 cluster_key_counts
        if cluster_key_counts is None:
            cluster_key_counts = self.cluster_key_counts_gpu[layer_idx]  # [bs, kv_heads, B, num_clusters]
        # Step 3: 计算各 tier 的 key 数量阈值
        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
        tier_counts = [int(math.ceil(kv_len * collision_ratio * ratio)) for ratio in tier_ratios]
        # Step 4: 调用融合的 CUDA kernel
        # codebook: [bs, heads, B, kv_len] - 不需要 transpose
        cache_cnt = self._update_cache_cnt_fused(
            sorted_cluster_ids,
            cluster_key_counts,
            codebook,  # [bs, heads, B, kv_len] uint8
            tier_counts
        )
        return cache_cnt


    
    def _coarse_filter_truncate(self, cache_cnt, candidate_ratio, layer_idx=None):
        """
        bs打到14，空隙可能消失，gpu利用率可能会升高
        简化版碰撞粗筛：直接按碰撞计数从高到低排序，取前 candidate_len 个
        Args:
            cache_cnt: [bs, kv_heads, kv_len] 碰撞计数 (值域 [0, 96])
            candidate_ratio: 候选集比例
            layer_idx: 当前层索引
        Returns:
            coarse_indices: [bs, kv_heads, candidate_len] 粗筛后的候选索引
            candidate_len: int - 候选数量
        """
        kv_len = cache_cnt.shape[-1]
        candidate_len = max(1, int(math.ceil(kv_len * candidate_ratio)))
        coarse_indices = self._radix_topk_ext.radix_topk_3d(cache_cnt, candidate_len)    
        return coarse_indices, candidate_len
    

    def _get_adaptive_ratios(self, retrieval_len):
        """
        根据 retrieval zone 长度自适应调节 candidate_ratio 和 collision_ratio
        Args:
            retrieval_len: retrieval zone 的长度
        Returns:
            (candidate_ratio, collision_ratio) 元组
        """
        if retrieval_len < 5000:
            candidate_ratio, collision_ratio = 0.5, 0.5   #(0, 5k)
        elif retrieval_len < 10000:
            candidate_ratio, collision_ratio = 0.25, 0.3  #(5k, 10k)
        elif retrieval_len < 30000:
            candidate_ratio, collision_ratio = 0.2, 0.25  #(10k, 30k)   
        elif retrieval_len < 100000:
            # candidate_ratio, collision_ratio = 0.2, 0.2 
            candidate_ratio, collision_ratio = 0.05, 0.2  #(30k, 100k)   short,medium,
        else:
            # (100k+) 超长样本：沿用较保守的设置，避免候选集过大
            candidate_ratio, collision_ratio = 0.05, 0.2
        return candidate_ratio, collision_ratio  

    
    def sync(self, layer_idx, batch_idx):
        """
        GPU版本:数据已经在prefill阶段直接存储到GPU,无需额外同步
        """
        pass  # GPU版本中所有数据已在prefill时存储到GPU,无需额外操作


