/**
 * Top-K kernels for KV cache collision counts
 * 
 * Two implementations:
 * 1. bucket_topk: Optimized for integer values in small range [0, MAX_VAL]
 *    - Uses counting sort approach: O(n + range)
 *    - Perfect for collision counts [0, 96]
 * 
 * 2. radix_topk: General-purpose float values
 *    - Uses radix select: O(n)
 *    - Better for arbitrary float distributions
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int MAX_BUCKET = 128;  // Support values 0-127

// =====================================================================
// Bucket TopK: Optimized for small integer range [0, MAX_VAL]
// =====================================================================

/**
 * Bucket Top-K kernel for integer values in [0, MAX_BUCKET-1]
 * 
 * Algorithm:
 * 1. Build histogram of values (256 buckets)
 * 2. Scan from highest bucket to find threshold
 * 3. Collect indices from buckets >= threshold
 * 
 * Much faster and more reliable than radix select for small integer ranges.
 */
__global__ void bucket_topk_kernel(
    const int32_t* __restrict__ input,  // [B, L]
    int64_t* __restrict__ output,        // [B, k] - int64 for compatibility with rerank kernel
    int64_t input_stride,
    int32_t length,
    int32_t k
) {
    const int bid = blockIdx.x;
    const int32_t* in = input + bid * input_stride;
    int64_t* out = output + bid * k;
    
    // Shared memory for histogram
    __shared__ int s_hist[MAX_BUCKET];
    __shared__ int s_threshold;
    __shared__ int s_counter;
    __shared__ int s_above_threshold_count;
    
    const int tid = threadIdx.x;
    
    // Initialize histogram
    if (tid < MAX_BUCKET) {
        s_hist[tid] = 0;
    }
    if (tid == 0) {
        s_counter = 0;
        s_threshold = 0;
        s_above_threshold_count = 0;
    }
    __syncthreads();
    
    // Build histogram
    for (int i = tid; i < length; i += blockDim.x) {
        int val = in[i];
        if (val >= 0 && val < MAX_BUCKET) {
            atomicAdd(&s_hist[val], 1);
        }
    }
    __syncthreads();
    
    // Single thread finds threshold by scanning from high to low
    if (tid == 0) {
        int cumsum = 0;
        for (int bucket = MAX_BUCKET - 1; bucket >= 0; --bucket) {
            cumsum += s_hist[bucket];
            if (cumsum >= k) {
                s_threshold = bucket;
                s_above_threshold_count = cumsum - s_hist[bucket];
                break;
            }
        }
    }
    __syncthreads();
    
    int threshold = s_threshold;
    int above_count = s_above_threshold_count;
    
    // Phase 1: Collect all indices with value > threshold
    for (int i = tid; i < length; i += blockDim.x) {
        int val = in[i];
        if (val > threshold) {
            int pos = atomicAdd(&s_counter, 1);
            if (pos < k) {
                out[pos] = i;
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Collect indices with value == threshold until we have k
    int remaining = k - s_counter;
    if (remaining > 0) {
        for (int i = tid; i < length; i += blockDim.x) {
            int val = in[i];
            if (val == threshold) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) {
                    out[pos] = i;
                }
            }
        }
    }
}


// =====================================================================
// Radix TopK: General-purpose for float values
// =====================================================================

constexpr int RADIX = 256;
constexpr size_t kSmem = 32 * 1024 * sizeof(uint32_t);

struct RadixTopKParams {
    const float* __restrict__ input;
    int64_t* __restrict__ indices;  // int64 for compatibility with rerank kernel
    int64_t input_stride;
    int32_t length;
    int32_t k;
};

__device__ __forceinline__ uint32_t float_to_ordered_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint8_t float_to_uint8(float x) {
    // For small positive values, use direct scaling instead of half conversion
    // This gives better bucket distribution for values [0, 96]
    if (x >= 0 && x <= 127) {
        return static_cast<uint8_t>(x * 2);  // Scale to [0, 254]
    }
    // Fallback for other values
    uint32_t bits = float_to_ordered_uint32(x);
    return static_cast<uint8_t>(bits >> 24);
}

__device__ void naive_topk(int64_t* __restrict__ indices, int32_t length, int32_t k) {
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) {
        indices[i] = (i < length) ? i : 0;
    }
}

__global__ void radix_topk_kernel(RadixTopKParams params) {
    const int bid = blockIdx.x;
    const float* input = params.input + bid * params.input_stride;
    int64_t* indices = params.indices + bid * params.k;
    const int length = params.length;
    const int k = params.k;
    
    if (length <= k) {
        naive_topk(indices, length, k);
        return;
    }
    
    constexpr int BLOCK_SIZE = 1024;
    
    extern __shared__ int shared_mem[];
    int* s_histogram = shared_mem;
    
    __shared__ int s_counter;
    __shared__ int s_threshold;
    __shared__ int s_above_count;
    
    const int tid = threadIdx.x;
    
    // Initialize
    for (int i = tid; i < RADIX; i += BLOCK_SIZE) {
        s_histogram[i] = 0;
    }
    if (tid == 0) {
        s_counter = 0;
        s_threshold = RADIX - 1;
        s_above_count = 0;
    }
    __syncthreads();
    
    // Build histogram
    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
        uint8_t bin = float_to_uint8(input[idx]);
        atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();
    
    // Find threshold
    if (tid == 0) {
        int cumsum = 0;
        for (int bucket = RADIX - 1; bucket >= 0; --bucket) {
            cumsum += s_histogram[bucket];
            if (cumsum >= k) {
                s_threshold = bucket;
                s_above_count = cumsum - s_histogram[bucket];
                break;
            }
        }
    }
    __syncthreads();
    
    int threshold = s_threshold;
    
    // Collect indices with value > threshold
    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
        uint8_t bin = float_to_uint8(input[idx]);
        if (bin > threshold) {
            int pos = atomicAdd(&s_counter, 1);
            if (pos < k) indices[pos] = idx;
        }
    }
    __syncthreads();
    
    // Collect indices with value == threshold until we have k
    int remaining = k - s_counter;
    if (remaining > 0) {
        for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
            uint8_t bin = float_to_uint8(input[idx]);
            if (bin == threshold) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) indices[pos] = idx;
            }
        }
    }
}

}  // namespace


// =====================================================================
// Python Interface
// =====================================================================

/**
 * Bucket Top-K for integer values [0, 127]
 * 
 * Args:
 *   input: [B, L] int32 tensor
 *   k: number of top elements
 * 
 * Returns:
 *   indices: [B, k] int32 tensor
 */
torch::Tensor bucket_topk_cuda(
    torch::Tensor input,
    int64_t k
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [B, L]");
    TORCH_CHECK(k > 0, "k must be positive");
    
    const int B = input.size(0);
    const int L = input.size(1);
    
    TORCH_CHECK(k <= L, "k cannot exceed sequence length");
    
    // Ensure int32 and contiguous
    torch::Tensor input_int;
    if (input.scalar_type() != torch::kInt32) {
        input_int = input.to(torch::kInt32).contiguous();
    } else {
        input_int = input.contiguous();
    }
    
    // Zeros (not empty): partial fills stay in-bounds for downstream gather; int64 for rerank.
    auto indices = torch::zeros({B, k}, 
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    
    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    bucket_topk_kernel<<<B, kThreadsPerBlock, 0, stream>>>(
        input_int.data_ptr<int32_t>(),
        indices.data_ptr<int64_t>(),
        input_int.stride(0),
        L,
        k
    );
    
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bucket_topk kernel failed: ", cudaGetErrorString(err));
    
    return indices;
}


/**
 * 3D wrapper: [bs, kv_heads, kv_len] -> [bs, kv_heads, k]
 */
torch::Tensor bucket_topk_3d_cuda(
    torch::Tensor input,
    int64_t k
) {
    TORCH_CHECK(input.dim() == 3, "input must be 3D [bs, kv_heads, kv_len]");
    
    const int bs = input.size(0);
    const int kv_heads = input.size(1);
    const int kv_len = input.size(2);
    
    auto input_2d = input.reshape({bs * kv_heads, kv_len});
    auto indices_2d = bucket_topk_cuda(input_2d, k);
    return indices_2d.reshape({bs, kv_heads, k});
}


/**
 * Radix Top-K for float values
 */
torch::Tensor radix_topk_cuda(
    torch::Tensor input,
    int64_t k
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [B, L]");
    TORCH_CHECK(k > 0, "k must be positive");
    
    const int B = input.size(0);
    const int L = input.size(1);
    
    TORCH_CHECK(k <= L, "k cannot exceed sequence length");
    
    torch::Tensor input_float;
    if (input.scalar_type() != torch::kFloat32) {
        input_float = input.to(torch::kFloat32).contiguous();
    } else {
        input_float = input.contiguous();
    }
    
    // Zeros (not empty): partial fills stay in-bounds for downstream gather; int64 for rerank.
    auto indices = torch::zeros({B, k}, 
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    
    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    RadixTopKParams params;
    params.input = input_float.data_ptr<float>();
    params.indices = indices.data_ptr<int64_t>();
    params.input_stride = input_float.stride(0);
    params.length = L;
    params.k = k;
    
    constexpr int BLOCK_SIZE = 1024;
    size_t smem = RADIX * sizeof(int);
    
    radix_topk_kernel<<<B, BLOCK_SIZE, smem, stream>>>(params);
    
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "radix_topk kernel failed: ", cudaGetErrorString(err));
    
    return indices;
}


torch::Tensor radix_topk_3d_cuda(
    torch::Tensor input,
    int64_t k
) {
    TORCH_CHECK(input.dim() == 3, "input must be 3D [bs, kv_heads, kv_len]");
    
    const int bs = input.size(0);
    const int kv_heads = input.size(1);
    const int kv_len = input.size(2);
    
    auto input_2d = input.reshape({bs * kv_heads, kv_len});
    auto indices_2d = radix_topk_cuda(input_2d, k);
    return indices_2d.reshape({bs, kv_heads, k});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bucket TopK (optimized for int values [0, 127])
    m.def("bucket_topk", &bucket_topk_cuda,
          "Bucket Top-K for integer values [0, 127] (2D input)",
          py::arg("input"), py::arg("k"));
    m.def("bucket_topk_3d", &bucket_topk_3d_cuda,
          "Bucket Top-K for integer values [0, 127] (3D input)",
          py::arg("input"), py::arg("k"));
    
    // Radix TopK (general-purpose)
    m.def("radix_topk", &radix_topk_cuda,
          "Radix Select Top-K (2D input)",
          py::arg("input"), py::arg("k"));
    m.def("radix_topk_3d", &radix_topk_3d_cuda,
          "Radix Select Top-K (3D input)",
          py::arg("input"), py::arg("k"));
}
