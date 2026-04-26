/*
 * adaptive_topk.cu — Fused adaptive top-k selection kernel
 *
 * 融合的操作（适合融合：数据依赖链，避免多次 kernel launch + 中间 tensor 分配）:
 *   global_max → exp → benefit → sum → radix_select → bitonic_sort → cumsum → adaptive_k → gather
 *
 * 不融合的操作（交给 cuBLAS / PyTorch，它们做得更好）:
 *   QK matmul (sl_logits = einsum(q, slk) * scale)  — tensor core 优化
 *   cand_logits = cand_scores.float() * attn_scale   — trivial element-wise
 *
 * 接口:
 *   输入: sl_logits [bs, H, sl]  fp32  (已乘 attn_scale 的精确 QK 内积)
 *         cand_logits [bs, H, C]  fp32  (已乘 attn_scale 的 RaBitQ 近似分数)
 *         sl_vnorm, cand_vnorm, cand_indices  (同原始接口)
 *   输出: adaptive_k [bs, H], topk_indices [bs, H, K_out]
 *
 * Grid:  (bs * H,)  — one block per (b, h)
 * Block: 512 threads
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

// ============================================================================
// Warp / Block reductions
// ============================================================================
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}
__device__ __forceinline__ int warp_sum_i(int v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

// All broadcast result to every thread via smem[0]
__device__ float blk_max(float v, float* s, int tid, int nt) {
    v = warp_max(v);
    if (tid % 32 == 0) s[tid / 32] = v;
    __syncthreads();
    v = (tid < (nt + 31) / 32) ? s[tid] : -FLT_MAX;
    if (tid / 32 == 0) v = warp_max(v);
    if (tid == 0) s[0] = v;
    __syncthreads(); v = s[0]; __syncthreads();
    return v;
}
__device__ float blk_sum(float v, float* s, int tid, int nt) {
    v = warp_sum(v);
    if (tid % 32 == 0) s[tid / 32] = v;
    __syncthreads();
    v = (tid < (nt + 31) / 32) ? s[tid] : 0.f;
    if (tid / 32 == 0) v = warp_sum(v);
    if (tid == 0) s[0] = v;
    __syncthreads(); v = s[0]; __syncthreads();
    return v;
}
__device__ int blk_sum_i(int v, int* s, int tid, int nt) {
    v = warp_sum_i(v);
    if (tid % 32 == 0) s[tid / 32] = v;
    __syncthreads();
    v = (tid < (nt + 31) / 32) ? s[tid] : 0;
    if (tid / 32 == 0) v = warp_sum_i(v);
    if (tid == 0) s[0] = v;
    __syncthreads(); v = s[0]; __syncthreads();
    return v;
}

// ============================================================================
// Bitonic sort (descending) in shared memory on (key, val) pairs
// n must be power of 2
// ============================================================================
__device__ void bitonic_sort_desc(float* k, int* v, int n, int tid, int nt) {
    for (int sz = 2; sz <= n; sz <<= 1) {
        for (int st = sz >> 1; st; st >>= 1) {
            for (int i = tid; i < n / 2; i += nt) {
                int g = i / st, p = i % st;
                int l = g * 2 * st + p, r = l + st;
                bool desc = ((l / sz) & 1) == 0;
                float lk = k[l], rk = k[r];
                bool swap = desc ? (lk < rk) : (lk > rk);
                if (swap) {
                    k[l] = rk; k[r] = lk;
                    int t = v[l]; v[l] = v[r]; v[r] = t;
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Main kernel: benefit_compute + radix_select + sort + adaptive_k + gather
//
// 输入已经是 fp32 logits（PyTorch 侧做好 matmul 和 scale）
// ============================================================================
__global__ void adaptive_topk_kernel(
    const float*   __restrict__ sl_logits,      // [bs, H, sl]  fp32, 已乘 scale
    const float*   __restrict__ cand_logits,    // [bs, H, C]   fp32, 已乘 scale
    const float*   __restrict__ sl_vnorm,       // [bs, H, sl]  fp32
    const int64_t* __restrict__ cand_indices,   // [bs, H, C]   int64
    const float*   __restrict__ cand_vnorm,     // [bs, H, C]   fp32
    float threshold,
    int K_out, int K_out_pad,
    int64_t* __restrict__ ak_out,               // [bs, H]
    int64_t* __restrict__ topk_idx_out,         // [bs, H, K_out]
    int bs, int H, int sl, int C,
    float* __restrict__ ws_ben                  // [bs*H, C] workspace
) {
    const int bh = blockIdx.x;
    const int b  = bh / H, h = bh % H;
    const int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ char raw[];
    float* rsm = (float*)raw;               // 32 floats (reduction scratch)
    float* sk  = rsm + 32;                  // K_out_pad floats (sort keys)
    int*   sv  = (int*)(sk + K_out_pad);    // K_out_pad ints  (sort vals)

    // Pointers for this (b, h)
    auto sll = sl_logits   + (int64_t)b*H*sl + (int64_t)h*sl;
    auto slv = sl_vnorm    + (int64_t)b*H*sl + (int64_t)h*sl;
    auto cl  = cand_logits + (int64_t)b*H*C  + (int64_t)h*C;
    auto cv  = cand_vnorm  + (int64_t)b*H*C  + (int64_t)h*C;
    auto ci  = cand_indices+ (int64_t)b*H*C  + (int64_t)h*C;
    float* ben = ws_ben    + (int64_t)bh*C;

    // ================================================================
    // Step 1: Joint global max across sl_logits and cand_logits
    // ================================================================
    float lmax = -FLT_MAX;
    for (int s = tid; s < sl; s += nt)
        lmax = fmaxf(lmax, sll[s]);
    for (int c = tid; c < C; c += nt)
        lmax = fmaxf(lmax, cl[c]);
    float gmax = blk_max(lmax, rsm, tid, nt);

    // ================================================================
    // Step 2: sl_benefit = sum( exp(sl_logits - gmax) * sl_vnorm )
    // ================================================================
    float lsl = 0.f;
    for (int s = tid; s < sl; s += nt)
        lsl += expf(sll[s] - gmax) * slv[s];
    float sl_benefit = blk_sum(lsl, rsm, tid, nt);

    // ================================================================
    // Step 3: cand_benefit = exp(cand_logits - gmax) * cand_vnorm
    //         → write to workspace; also reduce sum_all_cand
    // ================================================================
    float lcsum = 0.f;
    for (int c = tid; c < C; c += nt) {
        float v = expf(cl[c] - gmax) * cv[c];
        ben[c] = v;
        lcsum += v;
    }
    float sum_all = blk_sum(lcsum, rsm, tid, nt);
    float total = sl_benefit + sum_all;

    // ================================================================
    // Step 4: Radix select — find the K_out-th largest benefit
    //
    // benefit >= 0 → __float_as_uint is monotone.
    // Bit 31 (sign) is always 0, so start from bit 30.
    // MSB-first: at each bit, count how many remaining candidates
    // have that bit=1. If count >= remaining_k → require bit=1,
    // else consume all "1" items and continue with "0" branch.
    // After all bits, pivot = exact uint repr of the K_out-th value.
    // ================================================================
    unsigned mask = 0u, val = 0u;
    int rem = K_out;

    for (int bit = 30; bit >= 0 && rem > 0; bit--) {
        unsigned bm = 1u << bit;
        int cnt = 0;
        for (int c = tid; c < C; c += nt) {
            unsigned u = __float_as_uint(ben[c]);
            if ((u & mask) == val && (u & bm)) cnt++;
        }
        int tot = blk_sum_i(cnt, (int*)rsm, tid, nt);
        if (tot >= rem) {
            mask |= bm; val |= bm;   // require bit=1
        } else {
            mask |= bm;              // require bit=0
            rem -= tot;              // consume all "1" items
        }
    }
    unsigned pivot = val;

    // ================================================================
    // Step 5: Gather top-K_out into shared memory sort buffers
    //         - Pass 1: items with uint(benefit) > pivot
    //         - Pass 2: items with uint(benefit) == pivot (fill remaining)
    // ================================================================
    __shared__ int counter;
    if (tid == 0) counter = 0;
    for (int i = tid; i < K_out_pad; i += nt) { sk[i] = -1.f; sv[i] = 0; }
    __syncthreads();

    for (int c = tid; c < C; c += nt) {
        if (__float_as_uint(ben[c]) > pivot) {
            int slot = atomicAdd(&counter, 1);
            if (slot < K_out) { sk[slot] = ben[c]; sv[slot] = c; }
        }
    }
    __syncthreads();

    for (int c = tid; c < C; c += nt) {
        if (__float_as_uint(ben[c]) == pivot) {
            int slot = atomicAdd(&counter, 1);
            if (slot < K_out) { sk[slot] = ben[c]; sv[slot] = c; }
        }
    }
    __syncthreads();

    // Safety fallback: shouldn't happen, but fill if needed
    if (counter < K_out) {
        for (int c = tid; c < C; c += nt) {
            if (__float_as_uint(ben[c]) < pivot) {
                int slot = atomicAdd(&counter, 1);
                if (slot < K_out) { sk[slot] = ben[c]; sv[slot] = c; }
            }
        }
        __syncthreads();
    }

    // ================================================================
    // Step 6: Bitonic sort K_out_pad items descending
    // ================================================================
    bitonic_sort_desc(sk, sv, K_out_pad, tid, nt);

    // ================================================================
    // Step 7: Sequential cumsum → find adaptive_k
    // ================================================================
    if (tid == 0) {
        int fk = K_out;
        if (total <= 0.f) {
            fk = 1;
        } else {
            float cs = 0.f, tgt = threshold * total;
            for (int i = 0; i < K_out; i++) {
                cs += sk[i];
                if (sl_benefit + cs >= tgt) { fk = i + 1; break; }
            }
        }
        ak_out[bh] = (int64_t)fk;
    }

    // ================================================================
    // Step 8: Gather cand_indices by sort permutation → output
    // ================================================================
    auto out = topk_idx_out + (int64_t)bh * K_out;
    for (int i = tid; i < K_out; i += nt) {
        int oi = sv[i];
        out[i] = (oi >= 0 && oi < C) ? ci[oi] : 0;
    }
}


// ============================================================================
// Host launcher
// ============================================================================
void adaptive_topk(
    torch::Tensor sl_logits,       // [bs, H, sl]   fp32
    torch::Tensor cand_logits,     // [bs, H, C]    fp32
    torch::Tensor sl_vnorm,        // [bs, H, sl]   fp32
    torch::Tensor cand_indices,    // [bs, H, C]    int64
    torch::Tensor cand_vnorm,      // [bs, H, C]    fp32
    double threshold_d,
    int64_t K_out,
    torch::Tensor adaptive_k,      // [bs, H]       int64 (output)
    torch::Tensor topk_indices     // [bs, H, K_out] int64 (output)
) {
    int bs = sl_logits.size(0), H = sl_logits.size(1);
    int sl = sl_logits.size(2), C = cand_logits.size(2);
    int Ko = (int)K_out, Kp = 1;
    while (Kp < Ko) Kp <<= 1;

    // Workspace: cand_benefit array per (b, h)
    auto ws = torch::empty({bs * H, C},
        torch::TensorOptions().dtype(torch::kFloat32).device(sl_logits.device()));

    int grid = bs * H, block = 512;
    // smem: 32 (reduce) + Kp (sort_keys) + Kp (sort_vals as int)
    int smem = 32 * sizeof(float) + Kp * sizeof(float) + Kp * sizeof(int);

    adaptive_topk_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        sl_logits.data_ptr<float>(),
        cand_logits.data_ptr<float>(),
        sl_vnorm.data_ptr<float>(),
        cand_indices.data_ptr<int64_t>(),
        cand_vnorm.data_ptr<float>(),
        (float)threshold_d,
        Ko, Kp,
        adaptive_k.data_ptr<int64_t>(),
        topk_indices.data_ptr<int64_t>(),
        bs, H, sl, C,
        ws.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_topk", &adaptive_topk,
          "Fused adaptive top-k: benefit_compute + radix_select + sort + threshold + gather (CUDA)");
}
