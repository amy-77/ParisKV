#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <cuda_runtime.h>

/*
    coarse_indices.data_ptr(),  
    key_4bit.data_ptr(),    
    key_block_weight.data_ptr(),
    q_blocks.data_ptr(),
    q_norm.data_ptr(),
    mag_centers.data_ptr(),
    approx_scores.data_ptr(),
*/

struct RerankParams {
    const void *__restrict__ coarse_indices_ptr; // [bs, kv_heads, candidate_len] int64
    const void *__restrict__ key_4bit_ptr;     // [bs, kv_heads, B, kv_len, 4] uint8; last dim contiguous
    const void *__restrict__ key_block_weight_ptr;   // [bs, kv_heads, kv_len, B] - uint8
    const void *__restrict__ q_blocks_ptr;
    const void *__restrict__ q_norm_ptr;
    const void *__restrict__ mag_centers_ptr;
    void *__restrict__ approx_scores_ptr; // result: [bs, kv_heads, candidate_len]
    void *__restrict__ v4b_ptr;           // result: [bs, kv_heads, B, candidate_len, m] - unpacked 4bit values
    int m;
    int candidate_len;
    int kv_len;

    int key4_bs_stride;
    int key4_head_stride;
    int key4_B_stride;
    int key4_len_stride;

    int keyw_bs_stride;
    int keyw_head_stride;
    int keyw_B_stride;
};


template<int B>
__global__
void partial_rerank_kernel(
    __constant__ RerankParams params
) {
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;
    const int bid = (bidx * blockDim.y + bidy) * B;
    const int tidx = threadIdx.x;
    extern __shared__ uint8_t* smem_buffer;
    const int candidate_len = params.candidate_len;
    // const int kv_len = params.kv_len;
    const int m = params.m;
    const int key4_B_stride = params.key4_B_stride;
    const int key4_len_stride = params.key4_len_stride;
    const int keyw_B_stride = params.keyw_B_stride;

    const int64_t* coarse_indices_ptr = reinterpret_cast<const int64_t*>(params.coarse_indices_ptr) + (bidx * blockDim.y + bidy) * candidate_len;
    // load cand_4bit && unpack to register && save to shared memory
    // const uint32_t* cand_4bit_ptr = reinterpret_cast<const uint32_t*>(params.key_4bit_ptr) + bid * kv_len;
    const uint32_t* cand_4bit_ptr = reinterpret_cast<const uint32_t*>(params.key_4bit_ptr) + bidx * params.key4_bs_stride + bidy * params.key4_head_stride;
    const __nv_bfloat16* key_block_weight_ptr = reinterpret_cast<const __nv_bfloat16*>(params.key_block_weight_ptr) + bidx * params.keyw_bs_stride + bidy * params.keyw_head_stride;
    const __nv_bfloat16* q_blocks_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q_blocks_ptr) + bid * m;
    __nv_bfloat16* approx_scores_ptr = reinterpret_cast<__nv_bfloat16*>(params.approx_scores_ptr) + (bidx * blockDim.y + bidy) * candidate_len;
    // v4b output: [bs, kv_heads, B, candidate_len, m] - offset to current (bs, kv_head) position
    __nv_bfloat16* v4b_ptr = reinterpret_cast<__nv_bfloat16*>(params.v4b_ptr) + (bidx * blockDim.y + bidy) * B * candidate_len * m;
    __nv_bfloat16 q_norm = *(reinterpret_cast<const __nv_bfloat16*>(params.q_norm_ptr) + bidx * blockDim.y + bidy);

    __nv_bfloat16 mag_centers[8];
    *reinterpret_cast<uint4*>(&(mag_centers[0])) = __ldg(
        reinterpret_cast<const uint4*>(params.mag_centers_ptr)
    );

    __nv_bfloat16 res = __float2bfloat16(0.0f);
    // load cand_4bit
    const int global_id = bidz * blockDim.x + tidx;
    if (global_id < candidate_len){
        // load target index
        int64_t target_idx = __ldg(coarse_indices_ptr + global_id);

        #pragma unroll(4)
        for (int i = 0; i < B; i += 1){
            __nv_bfloat16 bit4_keys[8];
            __nv_bfloat16 q_blocks[8];
            // load q_blocks to register
            *reinterpret_cast<uint4*>(&(q_blocks[0])) = __ldg(reinterpret_cast<const uint4*>(q_blocks_ptr) + i);
            // load relavant 4byte vector
            uint32_t bit4_key = __ldg(cand_4bit_ptr + i * key4_B_stride + target_idx * key4_len_stride);
            // load weight
            __nv_bfloat16 bit4_weight = __ldg(key_block_weight_ptr + i * keyw_B_stride + target_idx);
            // unpack to register
            #pragma unroll
            for (int j = 0; j < 4 ; j++){
                const uint8_t even = static_cast<uint8_t>((bit4_key >> (j * 8)) & 0x0F);
                const uint8_t odd = static_cast<uint8_t>((bit4_key >> (j * 8 + 4)) & 0x0F);
                __nv_bfloat16 val = mag_centers[(even & 0x7)];
                if (even & 0x8){
                    bit4_keys[j * 2] = val;
                } else {
                    bit4_keys[j * 2] = -val;
                }

                val = mag_centers[odd & 0x7];
                if (odd & 0x8) {
                    bit4_keys[j * 2 + 1] = val;
                } else {
                    bit4_keys[j * 2 + 1] = -val;
                }
            }
            
            // Store unpacked v4b to global memory: v4b[bs, kv_heads, B, candidate_len, m]
            // Offset: block_i * candidate_len * m + global_id * m
            __nv_bfloat16* v4b_block_ptr = v4b_ptr + i * candidate_len * m + global_id * m;
            *reinterpret_cast<uint4*>(v4b_block_ptr) = *reinterpret_cast<uint4*>(&bit4_keys[0]);
            
            __nv_bfloat16 temp_res = __float2bfloat16(0.0f);
            #pragma unroll
            for (int j = 0; j < 8; j++){
                temp_res += bit4_keys[j] * q_blocks[j];
            }
            res += temp_res * bit4_weight;
        }
        res *= q_norm;
        // store
        approx_scores_ptr[global_id] = res;
    }
}


void partial_rerank_interface(
    at::Tensor coarse_indices,
    at::Tensor key_4bit,
    at::Tensor key_block_weight,
    at::Tensor q_blocks,
    at::Tensor q_norm,
    at::Tensor mag_centers,
    int B, int m, 
    at::Tensor approx_scores,
    at::Tensor v4b  // [bs, kv_heads, B, candidate_len, m] - unpacked 4bit values output
){

    // check dtype
    TORCH_CHECK(key_4bit.scalar_type() == at::ScalarType::Byte, 
                "key_4bit must be byte tensor");
    TORCH_CHECK(key_block_weight.scalar_type() == at::ScalarType::BFloat16, 
                "key_block_weight must be bfloat16 tensor");
    TORCH_CHECK(q_blocks.scalar_type() == at::ScalarType::BFloat16, 
                "q_blocks must be bfloat16 tensor");
    TORCH_CHECK(q_norm.scalar_type() == at::ScalarType::BFloat16, 
                "q_norm must be bfloat16 tensor");
    TORCH_CHECK(mag_centers.scalar_type() == at::ScalarType::BFloat16, 
                "mag_centers must be bfloat16 tensor");
    TORCH_CHECK(v4b.scalar_type() == at::ScalarType::BFloat16, 
                "v4b must be bfloat16 tensor");
    
    TORCH_CHECK(m == 8, "m must be 8");

    TORCH_CHECK(key_4bit.stride(-1) == 1, "key_4bit must be contiguous in the last dimension");
    TORCH_CHECK(key_block_weight.stride(-1) == 1, "key_block_weight must be contiguous in the last dimension");

    const unsigned int bs = key_4bit.size(0);
    const unsigned int kv_heads = key_4bit.size(1);
    const int kv_len = key_4bit.size(3);
    const int candidate_len = approx_scores.size(2);

    constexpr int BLOCK_SIZE = 128;
    const unsigned int block_num = (candidate_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // launch kernel
    const c10::Device dev = key_4bit.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid = dim3{bs, kv_heads, block_num};
    const auto block = dim3{BLOCK_SIZE};

    RerankParams params{
        coarse_indices.data_ptr(),  
        key_4bit.data_ptr(),    
        key_block_weight.data_ptr(),
        q_blocks.data_ptr(),
        q_norm.data_ptr(),
        mag_centers.data_ptr(),
        approx_scores.data_ptr(),
        v4b.data_ptr(),
        m, candidate_len, kv_len,
        // Strides in uint32_t words (byte stride / 4).
        int(key_4bit.stride(0) / 4), int(key_4bit.stride(1) / 4), int(key_4bit.stride(2) / 4), int(key_4bit.stride(3) / 4),
        int(key_block_weight.stride(0)), int(key_block_weight.stride(1)), int(key_block_weight.stride(2)),
    };

    int smem_size = 2 * (candidate_len);
    if (B == 16){
        cudaFuncSetAttribute(
            partial_rerank_kernel<16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);

        partial_rerank_kernel<16><<<grid, block, smem_size, stream>>>(
            params
        );
    } else {
        TORCH_CHECK(false, "only support B = 16");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "partial_rerank kernel failed:", ::cudaGetErrorString(result));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial_rerank", &partial_rerank_interface);
}