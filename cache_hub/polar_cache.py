"""
ParisKV: Polar ANN-based KV Cache with RaBitQ Reranking

A GPU-accelerated KV cache system for efficient long-context LLM inference.
Uses polar coordinate decomposition (R x Omega) for data-independent quantization
of keys, enabling fast approximate nearest neighbor retrieval during decoding.

Key components:
  - Prefill: Encodes keys using SRHT + sign-based product quantization
  - Decode: Multi-tier collision-based coarse retrieval + 4-bit RaBitQ reranking
  - Supports CPU offloading of retrieval-zone KV pairs for memory efficiency
"""
import math
import json
import gc
import os
import torch
from fast_hadamard_transform import hadamard_transform  # type: ignore
from cache_hub.rerank.rerank import partial_rerank_cuda

USE_RABITQ_RERANK = True
DEFAULT_SRHT_ROUNDS = 1

try:
    from .cache import KV_Cache
except ImportError:
    from cache import KV_Cache
from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat


class MultiDimBlockQuantizer:
    """
    Multi-dimensional block quantizer using polar (R x Omega) decomposition.
    Splits d-dimensional vectors into B blocks of m dimensions each,
    with direction quantized via sign-based spherical CVT codebooks.
    """

    def __init__(self, d, m, K_r, K_omega, codebook_path, seed=42, device=None):
        assert d % m == 0, f"Total dimension {d} must be divisible by block dimension {m}"
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")

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

        # Generate SRHT diagonal sign matrices with fixed seed for reproducibility
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)
        self.srht_diagonal_signs = []
        for _ in range(DEFAULT_SRHT_ROUNDS):
            signs = torch.randint(0, 2, (self.d,), device=device, dtype=torch.bfloat16) * 2 - 1
            self.srht_diagonal_signs.append(signs)

        # Restore RNG state to avoid affecting external sampling
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        # Load sign codebook to GPU
        try:
            with open(codebook_path, 'r') as f:
                codebook = json.load(f)

            config = codebook['config']
            if (config['d'] != self.d or config['m'] != self.m or
                config['K_r'] != self.K_r or config['K_omega'] != self.K_omega):
                raise ValueError(
                    f"Codebook parameter mismatch! "
                    f"Expected: d={self.d}, m={self.m}, K_r={self.K_r}, K_omega={self.K_omega}; "
                    f"Got: d={config['d']}, m={config['m']}, K_r={config['K_r']}, K_omega={config['K_omega']}"
                )

            if "angular_signs" in codebook:
                angular_signs = torch.tensor(codebook["angular_signs"], dtype=torch.int8, device=device)
                angular_signs = torch.where(angular_signs >= 0,
                    torch.ones_like(angular_signs), -torch.ones_like(angular_signs)).to(torch.int8)
            else:
                centers = torch.tensor(codebook['angular_centers'], dtype=torch.bfloat16, device=device)
                angular_signs = torch.where(centers >= 0,
                    torch.ones_like(centers), -torch.ones_like(centers)).to(torch.int8)

            self.V_omega_gpu = angular_signs.to(torch.bfloat16)

        except FileNotFoundError:
            raise FileNotFoundError(f"Codebook file not found: {codebook_path}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Codebook file format error: {e}")


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
        codebook_path: str = './turboquant/codebooks/codebook_d128_m8_Kr1_Kw256_rabitq_sign.json',
        final_topk: int = 100,
        dynamic_update_interval: int = 512,
        full_attention_threshold: int = 2000,
        enable_offload: bool = True,
    ) -> None:

        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads,
                         head_dim, dtype, layer_mapping, num_gpus, model_size)

        self.valid_start = valid_start
        self.sink_size = sink_size
        self.local_size = local_size
        self.steady_size = self.sink_size + self.local_size
        self.dynamic_update_interval = dynamic_update_interval if dynamic_update_interval is not None else self.local_size
        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head
        self.dtype = dtype
        self.max_new_length = max_new_length
        self.input_length = self.max_length - max_new_length
        self.FULL_ATTENTION_THRESHOLD = full_attention_threshold
        self.current_seq_len = [0] * layer_num
        if not os.path.isabs(codebook_path):
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            codebook_path = os.path.join(_project_root, codebook_path)
        self.codebook_path = codebook_path
        self.final_topk = final_topk
        self.enable_offload = enable_offload

        try:
            with open(codebook_path, 'r') as f:
                codebook_config = json.load(f)['config']
            self.polar_m = codebook_config['m']
            self.polar_K_r = codebook_config['K_r']
            self.polar_K_omega = codebook_config['K_omega']
            self.polar_B = head_dim // self.polar_m
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Cannot read codebook parameters: {e}")

        self._actual_retrieval_length = None
        self._decode_step_counter = 0
        self._buffer_base_counter = 0
        self._pending_buffer_reset = False

        import time as _time
        _t0 = _time.time()
        print(f"[Init] polar_cache (input_length={self.input_length}, max_new={max_new_length}, "
              f"layers={layer_num}, offload={self.enable_offload})")
        self.device = torch.device(list(layer_mapping.values())[0] if layer_mapping else 'cuda:0')
        self.quantizer = MultiDimBlockQuantizer(
            d=head_dim,
            m=self.polar_m,
            K_r=self.polar_K_r,
            K_omega=self.polar_K_omega,
            seed=42,
            codebook_path=codebook_path,
            device=self.device
        )

        self._V_omega_gpu_T = self.quantizer.V_omega_gpu.T.contiguous()
        self.unified_kv_capacity = self.input_length + self.max_new_length

        # Lloyd-Max 4-bit magnitude codebook (1 sign + 3-bit index, m=8).
        # Data-independent: derived from u_j^2 ~ Beta(1/2, (m-1)/2) on S^{m-1}.
        # Reproducible via: python run/generate_magnitude_levels.py --levels 8 --m 8
        self._mag_thresholds = torch.tensor(
            [0.0843, 0.1698, 0.2578, 0.3499, 0.4487, 0.5585, 0.6901],
            device=self.device, dtype=torch.bfloat16).contiguous()
        self._mag_centers = torch.tensor(
            [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
            device=self.device, dtype=torch.bfloat16).contiguous()
        self._bitpack_shifts = (1 << torch.arange(self.polar_m, device=self.device, dtype=torch.uint8))
        print(f"[Init] quantizer + codebook ready ({_time.time()-_t0:.1f}s)")

        if self.enable_offload:
            self.gpu_kv_capacity = self.sink_size + self.local_size + self.dynamic_update_interval
            print(f"[Init] Offload Mode: gpu_kv={self.gpu_kv_capacity}, "
                  f"cpu_kv={self.unified_kv_capacity - self.sink_size - self.local_size}")

            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._main_events = [torch.cuda.Event() for _ in range(layer_num)]
            self._copy_events = [torch.cuda.Event() for _ in range(layer_num)]
            with torch.cuda.device(self.device):
                for e in self._copy_events:
                    e.record()

            self._inflight_offload_buffers = [None for _ in range(layer_num)]
            self._layer_offload_inflight = [False for _ in range(layer_num)]
            self.cpu_kv_capacity = self.unified_kv_capacity - self.sink_size - self.local_size

            self.unified_keys_gpu = []
            self.unified_values_gpu = []
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim),
                                dtype=dtype, device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim),
                                dtype=dtype, device=self.device).contiguous())

            gc.collect()
            torch.cuda.empty_cache()

            self.unified_keys_cpu = []
            self.unified_values_cpu = []
            print(f"[Init] allocating CPU pinned KV ({layer_num} layers x {self.cpu_kv_capacity} tokens) ...")
            for ldx in range(layer_num):
                if ldx % 7 == 0:
                    gc.collect()
                self.unified_keys_cpu.append(
                    torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim),
                                dtype=dtype, pin_memory=True))
                self.unified_values_cpu.append(
                    torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim),
                                dtype=dtype, pin_memory=True))
            print(f"[Init] CPU pinned KV done ({_time.time()-_t0:.1f}s)")

            self.cpu_valid_end = [0] * layer_num
            max_topk_size = self.final_topk
            self._topk_keys_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim), dtype=dtype, device=self.device)
            self._topk_values_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim), dtype=dtype, device=self.device)

            self._offload_keys_buffer = [
                torch.empty((batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                            dtype=dtype, device=self.device)
                for _ in range(layer_num)
            ]
            self._offload_values_buffer = [
                torch.empty((batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                            dtype=dtype, device=self.device)
                for _ in range(layer_num)
            ]
            self._topk_keys_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),
                dtype=dtype, device='cpu', pin_memory=True)
            self._topk_values_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),
                dtype=dtype, device='cpu', pin_memory=True)

        else:
            self.gpu_kv_capacity = self.unified_kv_capacity
            print(f"[Init] No-Offload Mode: GPU capacity = {self.unified_kv_capacity} tokens")
            self.unified_keys_gpu = []
            self.unified_values_gpu = []
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.unified_kv_capacity, head_dim),
                                dtype=dtype, device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.unified_kv_capacity, head_dim),
                                dtype=dtype, device=self.device).contiguous())

            self.unified_keys_cpu = None
            self.unified_values_cpu = None
            self.cpu_valid_end = None
            self._topk_keys_buffer = None
            self._topk_values_buffer = None

        self.retrieval_zone_capacity = self.unified_kv_capacity
        print(f"[Init] allocating codebook/weight/4bit ({layer_num} layers x "
              f"{self.retrieval_zone_capacity} tokens) ...")

        # RaBitQ rerank weights (fused with key_norm): weight = (r / alpha) * ||k||
        self.key_block_weight_gpu = []
        for ldx in range(layer_num):
            self.key_block_weight_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity),
                            dtype=torch.bfloat16, device=self.device).contiguous())

        # 4-bit packed storage: 1-bit sign + 3-bit magnitude per dimension
        self._4bit_bytes_per_block = self.polar_m // 2
        self.key_4bit_packed_gpu = []
        for ldx in range(layer_num):
            self.key_4bit_packed_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity,
                             self._4bit_bytes_per_block),
                            dtype=torch.uint8, device=self.device).contiguous())

        self.codebook_gpu = [
            torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity),
                        dtype=torch.uint8, device=self.device)
            for _ in range(layer_num)
        ]
        self.cluster_key_counts_gpu = [
            torch.zeros((batch_size, self.kv_head, self.polar_B, self.polar_K_omega),
                        dtype=torch.int32, device=self.device)
            for _ in range(layer_num)
        ]
        self.context = 0
        self._preallocated_buffers = {}
        self._cache_cnt_buffer = None
        self._cluster_range_cache = None
        self._hadamard_scale = 1.0 / math.sqrt(self.head_dim)
        print(f"[Init] codebook/weight/4bit done ({_time.time()-_t0:.1f}s)")

        if not hasattr(polar_cache, '_threads_initialized'):
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            polar_cache._threads_initialized = True

        print(f"[Init] loading CUDA extensions ...")
        try:
            from .gather import load_gather_ext
            self._gather_ext = load_gather_ext()
        except Exception as e:
            print(f"[Init]   gather ext FAILED: {e}")
            self._gather_ext = None

        try:
            from .topk import load_radix_topk_ext
            self._radix_topk_ext = load_radix_topk_ext()
        except Exception as e:
            print(f"[Init]   radix_topk FAILED: {e}")
            self._radix_topk_ext = None

        try:
            from .collision.collision import update_cache_cnt_cuda
            self._update_cache_cnt_cuda = update_cache_cnt_cuda
        except Exception as e:
            raise RuntimeError(f"Collision CUDA kernel compile failed: {e}")

        try:
            from .collision_fused.collison_interface import (
                update_cache_cnt_cuda_interface,
                load_kernel_module as load_collision_module,
            )
            self._update_cache_cnt_fused = update_cache_cnt_cuda_interface
            load_collision_module("collision.cu", "update_cache_cnt")
        except Exception as e:
            raise RuntimeError(f"Collision Fused CUDA kernel compile failed: {e}")

        try:
            from .gather_trans.trans_h2d import h2d_gather_kv
            self._h2d_gather_kv = h2d_gather_kv
        except Exception as e:
            raise RuntimeError(f"H2D Gather CUDA kernel compile failed: {e}")

        try:
            from .rerank.rerank import load_kernel_module
            self._rerank_kernel = load_kernel_module("rerank.cu", "rerank")
        except Exception as e:
            raise RuntimeError(f"Rerank CUDA kernel compile failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Init] polar_cache.__init__ DONE ({_time.time()-_t0:.1f}s)")

    # =========================================================================
    # Prefill
    # =========================================================================

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx):
        """Prefill entry point. Dispatches to single-sample or lockstep multi-batch."""
        bsz = key_states.shape[0]
        if bsz == 1:
            return self._prefill_single_sample(key_states, value_states, layer_idx, batch_idx)
        if batch_idx != 0:
            vs = self.valid_start[0] if isinstance(self.valid_start, (list, tuple)) else int(self.valid_start)
            return key_states[:, vs:, :, :], value_states[:, vs:, :, :]
        return self._prefill_lockstep_batch(key_states, value_states, layer_idx)

    def _prefill_lockstep_batch(self, key_states, value_states, layer_idx: int):
        """Lockstep multi-batch prefill (assumes uniform valid_start across batch)."""
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == self.batch_size, (
            f"bsz({bsz}) must equal cache batch_size({self.batch_size}) for lockstep mode.")
        assert seq_len <= self.input_length

        if hasattr(self.valid_start, '__len__') and len(self.valid_start) > 1:
            assert len(self.valid_start) >= bsz
            vs0 = int(self.valid_start[0])
            for b in range(1, bsz):
                assert int(self.valid_start[b]) == vs0, (
                    f"lockstep multi-batch requires same valid_start: "
                    f"valid_start[0]={vs0}, valid_start[{b}]={self.valid_start[b]}")
            valid_start = vs0
        else:
            valid_start = int(self.valid_start) if not hasattr(self.valid_start, '__len__') else int(self.valid_start[0])

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
            if self.enable_offload:
                bs = self.unified_keys_cpu[0].shape[0]
                self._topk_keys_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_values_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_keys_cpu_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)

        # Short prefill: everything stays on GPU
        if valid_length == 0:
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            k_gpu = key_states[:, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            v_gpu = value_states[:, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :actual_len, :] = k_gpu
            self.unified_values_gpu[layer_idx][:bsz, :, :actual_len, :] = v_gpu
            self.current_seq_len[layer_idx] = actual_len
            if layer_idx == self.layer_num - 1:
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        # Step 1: Partition and store KV
        local_len = local_end - local_start
        if self.enable_offload:
            # Sink -> GPU
            sink_k = key_states[:, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
            sink_v = value_states[:, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :self.sink_size, :] = sink_k
            self.unified_values_gpu[layer_idx][:bsz, :, :self.sink_size, :] = sink_v

            # Local -> GPU
            local_k = key_states[:, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
            local_v = value_states[:, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, self.sink_size:self.sink_size+local_len, :] = local_k
            self.unified_values_gpu[layer_idx][:bsz, :, self.sink_size:self.sink_size+local_len, :] = local_v

            # Retrieval -> CPU (async D2H)
            temp_retrieval_keys = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            temp_retrieval_values = value_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
            self._layer_offload_inflight[layer_idx] = True
            self._main_events[layer_idx].record()

            with torch.cuda.stream(self._copy_stream):
                self._main_events[layer_idx].wait()
                self.unified_keys_cpu[layer_idx][:bsz, :, :retrieval_length, :].copy_(
                    temp_retrieval_keys, non_blocking=True)
                self.unified_values_cpu[layer_idx][:bsz, :, :retrieval_length, :].copy_(
                    temp_retrieval_values, non_blocking=True)
                self._copy_events[layer_idx].record()
            self.cpu_valid_end[layer_idx] = retrieval_length

        else:
            actual_len = seq_len - valid_start
            k_gpu = key_states[:, valid_start:seq_len, :, :].transpose(1, 2).to(self.dtype)
            v_gpu = value_states[:, valid_start:seq_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :actual_len, :] = k_gpu
            self.unified_values_gpu[layer_idx][:bsz, :, :actual_len, :] = v_gpu

        self.current_seq_len[layer_idx] = seq_len - valid_start

        # Step 2: Encode retrieval zone
        keys_to_encode = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).contiguous()
        codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
            keys_to_encode, layer_idx=layer_idx, return_rabitq=USE_RABITQ_RERANK)

        self.codebook_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end] = codebook_batch
        if USE_RABITQ_RERANK:
            self.key_block_weight_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end] = block_weight_batch
            self.key_4bit_packed_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end, :] = packed4b_batch

        # Step 3: Cluster counts
        self.cluster_key_counts_gpu[layer_idx][:bsz] = self._count_cluster_keys(
            codebook_batch, codebook_batch.device)

        if layer_idx == self.layer_num - 1:
            self.context += self.current_seq_len[layer_idx]
            if self.enable_offload:
                self._sync_offload_copies()
        else:
            if self.enable_offload:
                self._sync_layer_offload(layer_idx)

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

    def _prefill_single_sample(self, key_states, value_states, layer_idx, batch_idx):
        """Single-sample prefill: encode keys and partition into Sink/Local/Retrieval zones."""
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == 1, "Multi-batch prefilling only support single batch one by one."
        assert seq_len <= self.input_length
        valid_start = self.valid_start[batch_idx]

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
            if self.enable_offload:
                bs = self.unified_keys_cpu[0].shape[0]
                self._topk_keys_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_values_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_keys_cpu_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty(
                    (bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)

        # Short prefill: everything stays on GPU
        if valid_length == 0:
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            self.unified_keys_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                key_states[0, valid_start:valid_start+actual_len, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                value_states[0, valid_start:valid_start+actual_len, :, :].transpose(0, 1).to(self.dtype)
            self.current_seq_len[layer_idx] = actual_len
            if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        local_len = local_end - local_start

        if self.enable_offload:
            # Sink -> GPU
            self.unified_keys_gpu[layer_idx][batch_idx, :, :self.sink_size, :] = \
                key_states[0, sink_start:sink_end, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :self.sink_size, :] = \
                value_states[0, sink_start:sink_end, :, :].transpose(0, 1).to(self.dtype)

            # Local -> GPU
            self.unified_keys_gpu[layer_idx][batch_idx, :, self.sink_size:self.sink_size+local_len, :] = \
                key_states[0, local_start:local_end, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, self.sink_size:self.sink_size+local_len, :] = \
                value_states[0, local_start:local_end, :, :].transpose(0, 1).to(self.dtype)

            # Retrieval -> CPU (async D2H)
            temp_retrieval_keys = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).to(self.dtype)
            temp_retrieval_values = value_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).to(self.dtype)
            self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
            self._layer_offload_inflight[layer_idx] = True
            self._main_events[layer_idx].record()

            with torch.cuda.stream(self._copy_stream):
                self._main_events[layer_idx].wait()
                self.unified_keys_cpu[layer_idx][batch_idx, :, :retrieval_length, :].copy_(
                    temp_retrieval_keys, non_blocking=True)
                self.unified_values_cpu[layer_idx][batch_idx, :, :retrieval_length, :].copy_(
                    temp_retrieval_values, non_blocking=True)
                self._copy_events[layer_idx].record()

            self.cpu_valid_end[layer_idx] = retrieval_length

        else:
            actual_len = seq_len - valid_start
            self.unified_keys_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                key_states[0, valid_start:seq_len, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                value_states[0, valid_start:seq_len, :, :].transpose(0, 1).to(self.dtype)

        self.current_seq_len[layer_idx] = seq_len - valid_start

        # Encode retrieval zone
        keys_to_encode = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).contiguous().unsqueeze(0)
        codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
            keys_to_encode, layer_idx=layer_idx, return_rabitq=USE_RABITQ_RERANK)

        self.codebook_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end] = codebook_batch[0]
        if USE_RABITQ_RERANK:
            self.key_block_weight_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end] = block_weight_batch[0]
            self.key_4bit_packed_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end, :] = packed4b_batch[0]

        # Cluster counts
        self.cluster_key_counts_gpu[layer_idx][batch_idx] = self._count_cluster_keys(
            codebook_batch, codebook_batch.device)[0]

        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += self.current_seq_len[layer_idx]
            if self.enable_offload:
                self._sync_offload_copies()
        else:
            if self.enable_offload:
                self._sync_layer_offload(layer_idx)

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

    # =========================================================================
    # Offload synchronization
    # =========================================================================

    def _sync_offload_copies(self):
        """Wait for all async offload copies to complete."""
        if not self.enable_offload:
            return
        self._copy_stream.synchronize()
        if hasattr(self, "_inflight_offload_buffers"):
            for ldx in range(self.layer_num):
                self._inflight_offload_buffers[ldx] = None
                self._layer_offload_inflight[ldx] = False

    def _sync_layer_offload(self, layer_idx: int):
        """Wait for a single layer's async offload to complete."""
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

    # =========================================================================
    # Region index computation
    # =========================================================================

    def get_region_indices(self, layer_idx, device=None):
        """
        Compute dynamic region indices for the unified KV cache.

        Returns:
            dict with keys: sink_indices, local_indices, retrieval_start,
            retrieval_length, total_len, local_len
        """
        if device is None:
            device = self.device

        total_len = self.current_seq_len[layer_idx]
        steady_threshold = self.steady_size

        if self.input_length >= steady_threshold:
            update_buffer_size = max(0, total_len - self.input_length) % self.dynamic_update_interval
        elif total_len >= steady_threshold:
            update_buffer_size = (total_len - steady_threshold) % self.dynamic_update_interval
        else:
            update_buffer_size = 0

        local_end = max(total_len - update_buffer_size, self.sink_size)
        local_start = max(self.sink_size, local_end - self.local_size)
        retrieval_length = max(0, local_start - self.sink_size)

        sink_indices = torch.arange(0, self.sink_size, device=device, dtype=torch.long)
        local_len = total_len - local_start if local_start < total_len else 0
        if local_len > 0:
            local_indices = torch.arange(local_start, total_len, device=device, dtype=torch.long)
        else:
            local_indices = torch.zeros(0, device=device, dtype=torch.long)

        return {
            'sink_indices': sink_indices,
            'local_indices': local_indices,
            'retrieval_start': self.sink_size,
            'retrieval_length': retrieval_length,
            'total_len': total_len,
            'local_len': local_len,
        }

    def _compute_local_start(self, layer_idx=0):
        """Lightweight: compute local_start without building tensors."""
        total_len = self.current_seq_len[layer_idx]
        steady_threshold = self.steady_size

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
        """Gather full KV (Sink + Retrieval + Local) for attention, handling offload."""
        local_len = local_indices.shape[0]

        if self.enable_offload and self.unified_keys_cpu is not None and retrieval_length > 0:
            self._sync_layer_offload(layer_idx)

            sink_local_end = self.sink_size + local_len
            sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
            sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]

            cpu_retrieval_len = self.cpu_valid_end[layer_idx]
            if cpu_retrieval_len > 0:
                retrieval_keys = self.unified_keys_cpu[layer_idx][:, :, :cpu_retrieval_len, :].to(
                    device, non_blocking=True)
                retrieval_values = self.unified_values_cpu[layer_idx][:, :, :cpu_retrieval_len, :].to(
                    device, non_blocking=True)
                concat_keys = torch.cat([sink_local_keys, retrieval_keys], dim=2)
                concat_values = torch.cat([sink_local_values, retrieval_values], dim=2)
            else:
                concat_keys = sink_local_keys
                concat_values = sink_local_values

        else:
            if retrieval_length > 0:
                retrieval_end = retrieval_start + retrieval_length
                retrieval_indices = torch.arange(retrieval_start, retrieval_end, device=device, dtype=torch.long)
                all_indices = torch.cat([sink_indices, retrieval_indices, local_indices])
                concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
                concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)
            else:
                all_indices = torch.cat([sink_indices, local_indices]) if local_len > 0 else sink_indices
                concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
                concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)

        return concat_keys, concat_values

    # =========================================================================
    # Dynamic index update (encode new tokens from update buffer)
    # =========================================================================

    def _update_polar_index(self):
        """
        Periodic update: encode newly accumulated tokens in the update buffer
        into the retrieval zone, then slide the local window.
        """
        new_retrieval_end = self._compute_local_start(0)
        old_retrieval_end = self._actual_retrieval_length + self.sink_size
        encode_start = old_retrieval_end
        encode_end = new_retrieval_end
        num_new_tokens = encode_end - encode_start

        assert num_new_tokens > 0, f"num_new_tokens={num_new_tokens} <= 0"
        assert encode_start >= self.sink_size
        assert encode_end <= self.current_seq_len[0]

        self._actual_retrieval_length = encode_end - self.sink_size

        for ldx in range(self.layer_num):
            if self.enable_offload:
                offload_start = self.sink_size
                offload_end = self.sink_size + num_new_tokens
                new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, offload_start:offload_end, :]
            else:
                new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, encode_start:encode_end, :]

            codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
                new_tokens_to_encode, layer_idx=ldx, return_rabitq=USE_RABITQ_RERANK)
            self.codebook_gpu[ldx][:, :, :, encode_start:encode_end] = codebook_batch
            if USE_RABITQ_RERANK:
                self.key_block_weight_gpu[ldx][:, :, :, encode_start:encode_end] = block_weight_batch
                self.key_4bit_packed_gpu[ldx][:, :, :, encode_start:encode_end, :] = packed4b_batch
            cluster_counts = self._count_cluster_keys(codebook_batch, codebook_batch.device)
            self.cluster_key_counts_gpu[ldx] += cluster_counts
            del codebook_batch, block_weight_batch, packed4b_batch, cluster_counts

            if self.enable_offload and self.unified_keys_cpu is not None:
                cpu_write_start = self.cpu_valid_end[ldx]
                cpu_write_end = cpu_write_start + num_new_tokens

                # Use pre-allocated staging buffers to avoid allocation overhead
                src_keys = self._offload_keys_buffer[ldx][:, :, :num_new_tokens, :]
                src_values = self._offload_values_buffer[ldx][:, :, :num_new_tokens, :]
                src_keys.copy_(self.unified_keys_gpu[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens, :])
                src_values.copy_(self.unified_values_gpu[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens, :])
                self._inflight_offload_buffers[ldx] = (src_keys, src_values)
                self._layer_offload_inflight[ldx] = True

                self._main_events[ldx].record()
                with torch.cuda.stream(self._copy_stream):
                    self._main_events[ldx].wait()
                    dst_keys = self.unified_keys_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                    dst_values = self.unified_values_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                    dst_keys.copy_(src_keys, non_blocking=True)
                    dst_values.copy_(src_values, non_blocking=True)
                    self._copy_events[ldx].record()

                self.cpu_valid_end[ldx] = cpu_write_end

                # Slide GPU local window
                old_local_end = self.sink_size + self.local_size
                old_buffer_end = old_local_end + num_new_tokens
                new_local_start_src = self.sink_size + num_new_tokens

                # Clone needed when src/dst overlap (num_new_tokens < local_size)
                if num_new_tokens < self.local_size:
                    new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                    new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                else:
                    new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                    new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                self.unified_keys_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_keys
                self.unified_values_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_values
                if num_new_tokens < self.local_size:
                    del new_local_keys, new_local_values

        if self.enable_offload:
            self._sync_offload_copies()

        self.steady_size = self.sink_size + self.local_size

    # =========================================================================
    # Decode
    # =========================================================================

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        Decode: append new token to KV cache and trigger periodic index update
        when the update buffer is full.
        """
        key_to_add = key_states[:, 0, :, :]
        value_to_add = value_states[:, 0, :, :]

        if self.enable_offload:
            # GPU layout: [Sink][Local][UpdateBuffer]
            current_len = self.current_seq_len[layer_idx]
            steady_threshold = self.steady_size
            if current_len < steady_threshold:
                gpu_write_pos = current_len
            else:
                if self.input_length >= steady_threshold:
                    tokens_in_buffer = self._decode_step_counter - self._buffer_base_counter
                else:
                    tokens_in_buffer = current_len - steady_threshold
                buffer_offset = tokens_in_buffer % self.dynamic_update_interval
                gpu_write_pos = steady_threshold + buffer_offset

            self.unified_keys_gpu[layer_idx][:, :, gpu_write_pos, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, gpu_write_pos, :] = value_to_add

        else:
            current_len = self.current_seq_len[layer_idx]
            self.unified_keys_gpu[layer_idx][:, :, current_len, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, current_len, :] = value_to_add

        self.current_seq_len[layer_idx] += 1

        # Layer 0: track decode steps and trigger index update when buffer is full
        if layer_idx == 0:
            self._decode_step_counter += 1
            self.context += 1

            if self._pending_buffer_reset:
                self._buffer_base_counter = self._decode_step_counter - 1
                self._pending_buffer_reset = False

            current_total_len = self.current_seq_len[0]
            steady_threshold = self.steady_size

            if current_total_len > steady_threshold:
                if self.input_length >= steady_threshold:
                    buffer_tokens = self._decode_step_counter - self._buffer_base_counter
                else:
                    buffer_tokens = current_total_len - steady_threshold

                if buffer_tokens > 0 and buffer_tokens % self.dynamic_update_interval == 0:
                    self._update_polar_index()
                    self._pending_buffer_reset = True

        return None, None

    # =========================================================================
    # 4-bit RaBitQ encoding
    # =========================================================================

    def _pack_4bit_codes(self, signs: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """Pack 1-bit sign + 3-bit magnitude into 4-bit (2 values per byte)."""
        v4bit = (signs.to(torch.uint8) << 3) | magnitudes
        v_even = v4bit[..., 0::2]
        v_odd = v4bit[..., 1::2]
        packed = (v_odd << 4) | v_even
        return packed.to(torch.uint8).contiguous()

    def _unpack_4bit_codes(self, packed: torch.Tensor, m: int = 8) -> tuple:
        """Unpack 4-bit packed codes into sign (+-1 int8) and magnitude codes (0-7 int64)."""
        v_even = packed & 0x0F
        v_odd = (packed >> 4) & 0x0F
        v4bit = torch.stack([v_even, v_odd], dim=-1).view(*packed.shape[:-1], m)
        sign_bits = (v4bit >> 3) & 1
        s_sign = (2 * sign_bits - 1).to(torch.int8)
        t_codes = (v4bit & 0x7).to(torch.int64)
        return s_sign, t_codes

    def _encode_query_blocks(self, query: torch.Tensor) -> tuple:
        """
        Encode query into SRHT-transformed blocks for reranking.

        Args:
            query: [bs, kv_heads, head_dim]
        Returns:
            q_blocks: [bs, kv_heads, B, m]
            q_norm: [bs, kv_heads, 1]
        """
        q_norm = torch.linalg.vector_norm(query, ord=2, dim=-1, keepdim=True)
        query_normalized = query / q_norm.clamp(min=1e-8)
        Y = query_normalized
        for round_idx in range(DEFAULT_SRHT_ROUNDS):
            Y = Y * self.quantizer.srht_diagonal_signs[round_idx]
            Y = hadamard_transform(Y, scale=self._hadamard_scale)
        q_blocks = Y.view(query.shape[0], query.shape[1], self.polar_B, self.polar_m).contiguous()
        return q_blocks, q_norm

    def batch_encode(self, Y, layer_idx=None, return_rabitq: bool = True):
        """
        4-bit RaBitQ encoding: 1-bit sign + 3-bit magnitude.

        Args:
            Y: [bs, kv_heads, seq_len, head_dim] GPU tensor
            return_rabitq: if True, return full encoding; if False, only codebook

        Returns:
            codebook: [bs, kv_heads, B, seq_len] uint8 bitpacked signs
            block_weight: [bs, kv_heads, B, seq_len] bf16 (None if return_rabitq=False)
            packed_4bit: [bs, kv_heads, B, seq_len, m//2] uint8 (None if return_rabitq=False)
        """
        bs, kv_heads, seq_len, head_dim = Y.shape
        key_norm = torch.linalg.vector_norm(Y, ord=2, dim=-1, keepdim=True)
        Y_normalized = Y / key_norm.clamp(min=1e-8)
        key_norm = key_norm.squeeze(-1)

        Y_rotated = Y_normalized
        for round_idx in range(DEFAULT_SRHT_ROUNDS):
            Y_rotated = Y_rotated * self.quantizer.srht_diagonal_signs[round_idx]
            Y_rotated = hadamard_transform(Y_rotated, scale=self._hadamard_scale)

        Y_blocks = Y_rotated.view(bs, kv_heads, seq_len, self.quantizer.B, self.quantizer.m)

        # Direction quantization: bitpack sign(u) directly
        bits = (Y_blocks >= 0)
        angular_codes_u8 = (bits * self._bitpack_shifts).sum(dim=-1, dtype=torch.uint8)
        codebook = angular_codes_u8.permute(0, 1, 3, 2).contiguous()

        if not return_rabitq:
            return codebook, None, None

        # 4-bit RaBitQ: compute block radii and quantized directions
        block_radii = torch.linalg.vector_norm(Y_blocks, ord=2, dim=-1)
        denom = block_radii.unsqueeze(-1).clamp(min=1e-8)
        directions = Y_blocks / denom
        del Y_blocks, denom

        abs_u = directions.abs()
        del directions
        t_codes = torch.bucketize(abs_u, self._mag_thresholds)

        a_vals = self._mag_centers[t_codes]
        t_codes_u8 = t_codes.to(torch.uint8)
        del t_codes
        packed_4bit = self._pack_4bit_codes(bits, t_codes_u8)
        # alpha^(4b) = sum(a[t_j] * |u_j|)
        alpha4b = (a_vals * abs_u).sum(dim=-1, dtype=torch.float32)
        alpha4b = alpha4b.clamp(min=1e-6)
        # weight = (r / alpha) * ||k|| (RaBitQ formula)
        block_weight = (block_radii / alpha4b) * key_norm.unsqueeze(-1)
        block_weight = block_weight.permute(0, 1, 3, 2).to(torch.bfloat16).contiguous()
        packed_4bit = packed_4bit.permute(0, 1, 3, 2, 4).contiguous()
        return codebook, block_weight, packed_4bit

    def _count_cluster_keys(self, codebook, device):
        """Count keys per cluster (supports multi-batch)."""
        batch_size, kv_heads, B, kv_len = codebook.shape
        actual_K = self.polar_K_omega
        num_groups = kv_heads * B

        counts_list = []
        for b in range(batch_size):
            codebook_b = codebook[b]
            codebook_flat = codebook_b.reshape(-1).long()
            group_ids = torch.arange(num_groups, device=device).view(kv_heads, B, 1).expand(
                -1, -1, kv_len).reshape(-1)
            global_ids = codebook_flat + group_ids * actual_K
            counts_flat = torch.bincount(global_ids, minlength=num_groups * actual_K).to(torch.int32)
            counts_list.append(counts_flat.view(kv_heads, B, actual_K))

        return torch.stack(counts_list, dim=0)

    # =========================================================================
    # Decode attention
    # =========================================================================

    def compute(self, queries, layer_idx):
        """
        Decode attention for a single token.
        Uses full attention when sequence is short, otherwise switches to
        collision-based retrieval + RaBitQ reranking.
        """
        device = self.device
        region_info = self.get_region_indices(layer_idx, device=device)
        total_length = region_info['total_len']
        sink_indices = region_info['sink_indices']
        local_indices = region_info['local_indices']
        retrieval_start = region_info['retrieval_start']
        retrieval_length = region_info['retrieval_length']
        local_len = region_info['local_len']

        encoded_len = self._actual_retrieval_length or 0
        cpu_len = self.cpu_valid_end[layer_idx] if (
            self.enable_offload and self.cpu_valid_end is not None) else 0
        if self.enable_offload and self.cpu_valid_end is not None:
            valid_length = cpu_len
        else:
            valid_length = min(retrieval_length, encoded_len)
        retrieval_end_effective = retrieval_start + valid_length

        # Prepare query: [bs, 1, num_heads, head_dim] -> [bs, kv_heads, head_dim]
        bs, seqlen_q, head_q, head_dim = queries.shape
        query_group = queries[:, 0, :, :]
        queries_reshaped = query_group.view(bs, self.kv_head, self.group_size, self.head_dim)
        query_avg_gpu = queries_reshaped.mean(dim=2)
        query_batch = query_avg_gpu.unsqueeze(2)

        min_total_for_retrieval = max(
            self.FULL_ATTENTION_THRESHOLD,
            self.sink_size + self.local_size + self.final_topk * 2)
        use_full_attention = (total_length < min_total_for_retrieval) or (valid_length <= 0)

        if use_full_attention:
            concat_keys, concat_values = self._get_full_kv_for_attention(
                layer_idx, sink_indices, local_indices,
                retrieval_start, retrieval_length, device, queries.dtype)
            if layer_idx == 0 and not hasattr(self, '_full_attn_printed'):
                print(f"[Full Attention] threshold={min_total_for_retrieval}, "
                      f"total_len={total_length}, offload={self.enable_offload}")
                self._full_attn_printed = True
        else:
            if layer_idx == 0 and not hasattr(self, '_retrieval_printed'):
                print(f"[Retrieval Mode] total_len={total_length}, valid_len={valid_length}")
                self._retrieval_printed = True

            codebook_gpu = self.codebook_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective]
            key_block_weight_gpu = self.key_block_weight_gpu[layer_idx][
                :, :, :, retrieval_start:retrieval_end_effective]
            key_4bit_gpu = self.key_4bit_packed_gpu[layer_idx][
                :, :, :, retrieval_start:retrieval_end_effective, :]

            assert not self.enable_offload or retrieval_start == self.sink_size

            retrieval_len = valid_length
            adaptive_candidate_ratio, adaptive_collision_ratio = self._get_adaptive_ratios(retrieval_len)

            topk_indices_batch = self.collision_based_topk_batch(
                query=query_batch,
                codebook=codebook_gpu,
                key_block_weight=key_block_weight_gpu,
                key_4bit=key_4bit_gpu,
                collision_ratio=adaptive_collision_ratio,
                candidate_ratio=adaptive_candidate_ratio,
                layer_idx=layer_idx,
            )

            sink_indices_exp = sink_indices.unsqueeze(0).unsqueeze(0).expand(bs, self.kv_head, -1)
            local_indices_exp = local_indices.unsqueeze(0).unsqueeze(0).expand(bs, self.kv_head, -1)

            if self.enable_offload and self.unified_keys_cpu is not None:
                # UVA H2D Gather: GPU indices + pinned CPU src -> GPU dst
                self._h2d_gather_kv(
                    self.unified_keys_cpu[layer_idx],
                    self.unified_values_cpu[layer_idx],
                    topk_indices_batch,
                    self._topk_keys_buffer,
                    self._topk_values_buffer,
                )
                sink_local_end = self.sink_size + local_len
                sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
                sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]
                concat_keys = torch.cat([sink_local_keys, self._topk_keys_buffer], dim=2)
                concat_values = torch.cat([sink_local_values, self._topk_values_buffer], dim=2)
            else:
                topk_indices_abs = topk_indices_batch + retrieval_start
                all_indices = torch.cat([sink_indices_exp, topk_indices_abs, local_indices_exp], dim=2)
                all_indices_exp = all_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                concat_keys = torch.gather(self.unified_keys_gpu[layer_idx], 2, all_indices_exp)
                concat_values = torch.gather(self.unified_values_gpu[layer_idx], 2, all_indices_exp)
                del all_indices, all_indices_exp, topk_indices_abs

        # Flash Attention
        k_cache = concat_keys.transpose(1, 2)
        v_cache = concat_values.transpose(1, 2)
        attn_out = flash_attn_with_kvcache_compat(
            q=queries,
            k_cache=k_cache,
            v_cache=v_cache,
            causal=False,
        )
        del concat_keys, concat_values, k_cache, v_cache
        return attn_out

    # =========================================================================
    # Collision-based retrieval
    # =========================================================================

    def _get_adaptive_ratios(self, retrieval_len):
        """Adapt candidate_ratio and collision_ratio based on retrieval zone length."""
        if retrieval_len < 5000:
            return 0.5, 0.5
        elif retrieval_len < 10000:
            return 0.25, 0.3
        elif retrieval_len < 30000:
            return 0.2, 0.25
        elif retrieval_len < 100000:
            return 0.1, 0.2
        else:
            return 0.1, 0.2

    def collision_based_topk_batch(self, query, codebook, key_block_weight=None,
                                     key_4bit=None, collision_ratio=0.2,
                                     candidate_ratio=0.2, layer_idx=None):
        """
        Collision-based top-k retrieval with RaBitQ reranking.

        Args:
            query: [bs, kv_heads, 1, head_dim]
            codebook: [bs, kv_heads, B, kv_len]
            key_block_weight: [bs, kv_heads, B, kv_len] bf16
            key_4bit: [bs, kv_heads, B, kv_len, m//2] uint8
            collision_ratio: fraction of clusters to activate
            candidate_ratio: fraction of keys as candidates

        Returns:
            topk_indices: [bs, kv_heads, final_topk]
        """
        bs, kv_heads, qlen, head_dim = query.shape
        assert qlen == 1
        query_squeezed = query.squeeze(2)
        q_blocks, q_norm = self._encode_query_blocks(query_squeezed)

        cache_cnt = self.get_cache_cnt_fused(q_blocks, codebook, collision_ratio, layer_idx)

        if key_block_weight is not None and key_4bit is not None:
            topk_indices = self.get_candidate_cache_with_rabitq_rerank(
                cache_cnt=cache_cnt,
                candidate_ratio=candidate_ratio,
                q_blocks=q_blocks,
                key_4bit=key_4bit,
                key_block_weight=key_block_weight,
                q_norm=q_norm,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError("RaBitQ rerank requires key_block_weight and key_4bit")
        return topk_indices

    def get_topk_clusters(self, q_blocks, codebook, collision_ratio, layer_idx):
        """
        Select top clusters per subspace using 6-tier weighted scoring.

        Tier boundaries: [0-5%, 5-15%, 15-30%, 30-50%, 50-75%, 75-100%]
        with weights [6, 5, 4, 3, 2, 1] respectively.

        Returns:
            sorted_cluster_ids: [bs, kv_heads, B, K_omega]
            tier_cluster_cnts: tuple of 6 tensors [bs, kv_heads, B]
        """
        kv_len = codebook.shape[3]
        scores = torch.matmul(q_blocks, self._V_omega_gpu_T)
        sorted_cluster_ids = scores.argsort(dim=-1, descending=True)
        cluster_key_counts = self.cluster_key_counts_gpu[layer_idx]
        sorted_key_counts = torch.gather(cluster_key_counts, dim=-1, index=sorted_cluster_ids.long())
        cumsum_counts = torch.cumsum(sorted_key_counts, dim=-1)
        before_cumsum = cumsum_counts - sorted_key_counts

        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
        tier_cluster_cnts = []

        for ratio in tier_ratios:
            target_keys = math.ceil(kv_len * collision_ratio * ratio)
            valid_mask = before_cumsum < target_keys
            tier_cluster_cnts.append(valid_mask.sum(dim=-1))

        return sorted_cluster_ids, tuple(tier_cluster_cnts)

    def update_cache_cnt(self, sorted_cluster_ids, tier_cluster_cnts, codebook):
        """Compute collision counts using 6-tier weighted CUDA kernel."""
        if isinstance(tier_cluster_cnts, (list, tuple)):
            if len(tier_cluster_cnts) == 0:
                raise ValueError("tier_cluster_cnts cannot be empty")
            topk_cluster_cnt = tier_cluster_cnts[-1]
        else:
            topk_cluster_cnt = tier_cluster_cnts
        return self._update_cache_cnt_cuda(
            sorted_cluster_ids.contiguous(),
            topk_cluster_cnt.contiguous(),
            codebook.contiguous(),
        )

    def get_cache_cnt_fused(self, q_blocks, codebook, collision_ratio, layer_idx):
        """
        Fused: get_topk_clusters + update_cache_cnt in a single CUDA call.

        Computes query-cluster scores, sorts, determines tier thresholds,
        and counts collisions in one fused kernel.
        """
        kv_len = codebook.shape[3]

        scores = torch.matmul(q_blocks, self._V_omega_gpu_T)
        sorted_cluster_ids = scores.argsort(dim=-1, descending=True).to(torch.int64)

        cluster_key_counts = self.cluster_key_counts_gpu[layer_idx]

        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
        tier_counts = [int(math.ceil(kv_len * collision_ratio * ratio)) for ratio in tier_ratios]

        cache_cnt = self._update_cache_cnt_fused(
            sorted_cluster_ids,
            cluster_key_counts,
            codebook,
            tier_counts,
        )

        return cache_cnt

    # =========================================================================
    # Reranking
    # =========================================================================

    def _coarse_filter_truncate(self, cache_cnt, candidate_ratio, layer_idx=None):
        """Coarse filter: select top candidates by collision count using radix topk."""
        kv_len = cache_cnt.shape[-1]
        candidate_len = max(1, int(math.ceil(kv_len * candidate_ratio)))
        coarse_indices = self._radix_topk_ext.radix_topk_3d(cache_cnt, candidate_len)
        return coarse_indices, candidate_len

    def get_candidate_cache_with_rabitq_rerank(
        self, cache_cnt, candidate_ratio, q_blocks, key_4bit, key_block_weight,
        q_norm, layer_idx=None
    ):
        """
        4-bit RaBitQ approximate inner-product reranking.

        Pipeline: Coarse filter -> Gather+Unpack+Score (fused CUDA) -> Final top-k.

        Args:
            cache_cnt: [bs, kv_heads, kv_len] collision counts
            candidate_ratio: fraction of keys to include as candidates
            q_blocks: [bs, kv_heads, B, m] SRHT-transformed query blocks
            key_4bit: [bs, kv_heads, B, kv_len, m//2] 4-bit packed codes
            key_block_weight: [bs, kv_heads, B, kv_len] bf16 fused weights
            q_norm: [bs, kv_heads, 1] query norms

        Returns:
            final_indices: [bs, kv_heads, final_topk]
        """
        coarse_indices, max_candidates = self._coarse_filter_truncate(
            cache_cnt, candidate_ratio, layer_idx)

        if self.polar_B == 16 and self.polar_m == 8:
            approx_scores, v4b_all = partial_rerank_cuda(
                coarse_indices.contiguous(),
                key_4bit,
                key_block_weight,
                q_blocks.contiguous(),
                q_norm.contiguous(),
                self._mag_centers.contiguous(),
                self.polar_B,
                self.polar_m,
                kernel=self._rerank_kernel,
            )
        else:
            # PyTorch fallback for non-standard block dimensions
            idx_code = coarse_indices.unsqueeze(2).expand(-1, -1, self.polar_B, -1)
            cand_weight = torch.gather(key_block_weight, dim=3, index=idx_code)
            idx_4bit = coarse_indices.unsqueeze(2).unsqueeze(-1).expand(
                -1, -1, self.polar_B, -1, key_4bit.shape[-1])
            cand_4bit = torch.gather(key_4bit, dim=3, index=idx_4bit)

            orig_shape = cand_4bit.shape[:-1]
            s_sign, t_codes = self._unpack_4bit_codes(
                cand_4bit.view(-1, cand_4bit.shape[-1]), m=self.polar_m)
            s_sign = s_sign.view(*orig_shape, self.polar_m)
            t_codes = t_codes.view(*orig_shape, self.polar_m)
            a_vals = self._mag_centers[t_codes]

            v4b = a_vals * s_sign.to(a_vals.dtype)
            q_exp = q_blocks.unsqueeze(3)
            per_block_scores = (q_exp * v4b).sum(dim=-1)
            cosine_proxy = (per_block_scores * cand_weight).sum(dim=2)
            approx_scores = cosine_proxy * q_norm

        actual_topk = min(self.final_topk, approx_scores.shape[-1])
        _, rerank_indices = torch.topk(approx_scores, k=actual_topk, dim=-1)
        final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)

        return final_indices

    # =========================================================================
    # Misc
    # =========================================================================

    def sync(self, layer_idx, batch_idx):
        """No-op: all data is stored during prefill."""
        pass
