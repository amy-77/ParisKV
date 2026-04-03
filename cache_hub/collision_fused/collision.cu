#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <cuda_runtime.h>
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template<int TIER_NUM>
struct Params {

    const int64_t *__restrict__ sorted_cluster_ids_ptr;
    const int32_t *__restrict__ cluster_key_counts_ptr;
    uint32_t *__restrict__ tier_bits_ptr;
    const uint8_t *__restrict__ code_book_ptr;   // [bs, kv_heads, kv_len, B] - uint8
    int32_t *__restrict__ cache_cnt_ptr;
    uint32_t __restrict__ tier_counts[TIER_NUM]; // 6 tiers, each [bs, kv_heads, B, num_clusters/32]

    // Strides for non-contiguous tensors.
    int cd_bs_stride;
    int cd_head_stride;
    int cd_b_stride;
    int sci_bs_stride;
    int sci_head_stride;
    int sci_b_stride;
    int ckc_bs_stride;
    int ckc_head_stride;
    int ckc_b_stride;

    int batchsize;
    int kv_heads;
    int sub_spaces;
    int kv_len;
};


__device__ __forceinline__ 
void warp_prefix_sum_inplace(int& val, int tidx_in_warp) {
    int temp = __shfl_up_sync(0xffffffff, val, 1);
    if (tidx_in_warp >= 1) val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 2);
    if (tidx_in_warp >= 2) val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 4);
    if (tidx_in_warp >= 4) val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 8);
    if (tidx_in_warp >= 8) val += temp;
    temp = __shfl_up_sync(0xffffffff, val, 16);
    if (tidx_in_warp >= 16) val += temp;
}

template<int Bits>
struct __align__(4) uint_bits{
    static_assert(Bits > 0, "Bit width must be positive.");
    static constexpr int StorageWords = (Bits + 31) / 32;
    uint32_t data[StorageWords];
};





template<int NUM_CLUSTER, int TIER_NUM>
__global__ void convert_to_bits(__constant__ Params<TIER_NUM> params){

    const int warp_idx = threadIdx.x / 32;
    // const int warp_num = blockDim.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int batchsize = params.batchsize;
    const int kv_heads = params.kv_heads;
    const int sub_spaces = params.sub_spaces;
    const int batchidx = bidx / (kv_heads * sub_spaces);
    const int headidx = (bidx % (kv_heads * sub_spaces)) / sub_spaces;
    const int spaceidx = bidx % sub_spaces;
    const int32_t* cluster_key_counts_ptr = params.cluster_key_counts_ptr + batchidx * params.ckc_bs_stride + 
        headidx * params.ckc_head_stride + spaceidx * params.ckc_b_stride;
    const int64_t* cluster_ids_ptr = params.sorted_cluster_ids_ptr + batchidx * params.sci_bs_stride + 
        headidx * params.sci_head_stride + spaceidx * params.sci_b_stride;



    int index = static_cast<int>(__ldg(cluster_ids_ptr + tidx));
    int count_native = __ldg(cluster_key_counts_ptr + index);
    int count = count_native;
    warp_prefix_sum_inplace(count, tid_in_warp);
    __shared__ int count_warp[NUM_CLUSTER / 32];

    if (tid_in_warp == 31) {
        count_warp[warp_idx] = count;
    }
    __syncthreads();
    // Cross-warp prefix sum: warp 0 only (at most NUM_CLUSTER/32 warps).
    if( warp_idx == 0 ){
        int temp_count = 0;
        if (tid_in_warp <  NUM_CLUSTER / 32){
            temp_count = count_warp[tid_in_warp];
        }
        warp_prefix_sum_inplace(temp_count, tid_in_warp);
        if (tid_in_warp <  NUM_CLUSTER / 32){
            count_warp[tid_in_warp] = temp_count;
        }
    }
    __shared__ uint_bits<NUM_CLUSTER> bits_tiers[TIER_NUM];
    #pragma unroll
    for (int t = tidx; t < TIER_NUM * NUM_CLUSTER / 32; t+= blockDim.x) {
        bits_tiers[t / (NUM_CLUSTER / 32) ].data[t % (NUM_CLUSTER / 32)] = 0u;
    }



    
    __syncthreads();
    if (warp_idx > 0){
        count += count_warp[warp_idx - 1];
    }
    count -= count_native;






    #pragma unroll
    for ( int i = 0; i < TIER_NUM; i++){
        int target_num = params.tier_counts[i];
        if (count < target_num){
            atomicOr(&(bits_tiers[i].data[index / 32]), 1u << (index % 32));
        }
    }

    __syncthreads();

    uint32_t* tier_bits_ptr = params.tier_bits_ptr + bidx * NUM_CLUSTER / 32 * TIER_NUM;

    #pragma unroll
    for(int i = tidx; i < TIER_NUM * NUM_CLUSTER / 32; i+= blockDim.x){
        int tier_idx = i / (NUM_CLUSTER / 32);
        int cluster_idx = i % (NUM_CLUSTER / 32);
        tier_bits_ptr[i] = bits_tiers[tier_idx].data[cluster_idx];
    }
}











template<int SUB_SPACE, int NUM_CLUSTERS, int BLOCK_SIZE, int NUM_TIERS>
__global__
void update_cache_cnt_kernel(const Params<NUM_TIERS> params){

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int kv_len = params.kv_len;
    const int batchsize = params.batchsize;
    const int kv_heads = params.kv_heads;
    const int cd_b_stride = params.cd_b_stride;
    int* cache_cnt_ptr = params.cache_cnt_ptr + bidx * kv_len;
    
    const int batch_idx = blockIdx.x / kv_heads;
    const int head_idx = blockIdx.x % kv_heads;

    const uint8_t* code_book_ptr = params.code_book_ptr + batch_idx * params.cd_bs_stride + head_idx * params.cd_head_stride;

    // if (bidx == 0 && bidy == 0 && threadIdx.x == 0) {
    //     printf("code_book_ptr: %p %d %d %d\n", code_book_ptr, params.cd_bs_stride, params.cd_head_stride, cd_b_stride);
    // }

    int tidx = threadIdx.x;

    // shared memory for 6 tiers
    __shared__ uint_bits<NUM_CLUSTERS> tier_bits_smem[SUB_SPACE * NUM_TIERS];
    uint32_t* tier_bits_ptr = params.tier_bits_ptr + bidx * SUB_SPACE * NUM_CLUSTERS / 32 * NUM_TIERS;

    // int4 (128-bit) vector loads
    #pragma unroll
    for (int i = tidx; i < SUB_SPACE * NUM_CLUSTERS / 32 * NUM_TIERS / 4; i += blockDim.x) {
        const int4* gptr = reinterpret_cast<const int4*>(tier_bits_ptr) + i;
        int4* sptr = reinterpret_cast<int4*>(tier_bits_smem) + i;
        *sptr = __ldg(gptr);
    }

    __syncthreads();

    const int start_k = bidy * BLOCK_SIZE;
    const int end_k = min(kv_len, start_k + BLOCK_SIZE);

    #pragma unroll
    for (int ks = tidx + start_k; ks < end_k; ks += blockDim.x) {
        int cnt = 0;
        #pragma unroll
        for(int i = 0; i < SUB_SPACE; i += 1) {
            // const int target_cluster_id = static_cast<int>(code_book_ptr[i * cd_b_stride + ks]);
            int target_cluster_id = static_cast<int>(code_book_ptr[i * cd_b_stride + ks]);
            // int target_cluster_id = 0;
            // const int idx_0 = target_cluster_id >> 5;
            // const uint32_t mask = 1u << (target_cluster_id & 0x1F);
            const int idx_0 = target_cluster_id / 32;
            const uint32_t mask = 1u << (target_cluster_id % 32);

            // Tiers are cumulative bitsets; count hits across tiers.
            #pragma unroll
            for (int t = 0; t < NUM_TIERS; t++) {
                bool hit = tier_bits_smem[ i * NUM_TIERS + t ].data[idx_0] & mask;
                // bool hit = tier_bits_smem[ 0 ].data[idx_0] & mask;
                if (hit){
                    cnt += 1;
                }
            }
        }
        __syncwarp();
        cache_cnt_ptr[ks] = cnt;
    }
}





template<int SUB_SPACES, int NUM_CLUSTERS, int BLOCK_SIZE, int TIER_NUM>
inline void launch_update_cache_cnt(dim3 grid, dim3 block, int smem_size,
                             const Params<TIER_NUM> &params)
{
    cudaFuncSetAttribute(
        update_cache_cnt_kernel<SUB_SPACES, NUM_CLUSTERS, BLOCK_SIZE, TIER_NUM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    update_cache_cnt_kernel<SUB_SPACES, NUM_CLUSTERS, BLOCK_SIZE, TIER_NUM><<<grid, block, smem_size>>>(params);
}




template<int TIER_NUM>
void update_cache_cnt_tier_xx(
    at::Tensor& sorted_cluster_ids, // [bs, kv_heads, B, num_clusters] int64
    at::Tensor& cluster_key_counts, // [bs, kv_heads, B, num_clusters] keys per cluster
    at::Tensor& code_book,          // [bs, kv_heads, B, kv_len] cluster id per key
    const std::vector<uint32_t>& tier_counts, // cumulative key-count thresholds per tier, e.g. [15,45,90,150,225,300]
    at::Tensor& cache_cnt           // [bs, kv_heads, kv_len] collision count per key
) {
    /*
     * Six-tier weighted collision counts (tier_ratios ~ [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]).
     * tier_counts entries are cumulative thresholds.
     */

    // check dtype
    TORCH_CHECK(code_book.scalar_type() == at::ScalarType::Byte, 
                "code_book must be uint8 tensor");
    TORCH_CHECK(sorted_cluster_ids.scalar_type() == at::ScalarType::Long,
                "sorted_cluster_ids must be int64 tensor");
    TORCH_CHECK(cluster_key_counts.scalar_type() == at::ScalarType::Int,
                "cluster_key_counts must be int32 tensor");
    // check shape
    TORCH_CHECK(sorted_cluster_ids.dim() == 4);
    TORCH_CHECK(code_book.dim() == 4);

    const int batchsize = sorted_cluster_ids.size(0);
    const int kv_heads = sorted_cluster_ids.size(1);
    const int sub_spaces = sorted_cluster_ids.size(2);
    const int num_clusters = sorted_cluster_ids.size(3);
    const int kv_len = code_book.size(3);
    TORCH_CHECK(num_clusters % 32 == 0, "num_clusters must be a multiple of 32");
    // launch kernel
    const c10::Device dev = cache_cnt.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto tier_bits_tensor = torch::zeros({batchsize, kv_heads, sub_spaces, TIER_NUM * num_clusters / 32},
                                             sorted_cluster_ids.options().dtype(torch::kUInt32));
    Params<TIER_NUM> params;
    params.sorted_cluster_ids_ptr = sorted_cluster_ids.data_ptr<int64_t>();
    params.cluster_key_counts_ptr = cluster_key_counts.data_ptr<int32_t>();
    params.tier_bits_ptr = tier_bits_tensor.data_ptr<uint32_t>();
    params.code_book_ptr = code_book.data_ptr<uint8_t>();
    params.cache_cnt_ptr = cache_cnt.data_ptr<int32_t>();

    for (int i = 0; i < TIER_NUM; i++) {
        params.tier_counts[i] = tier_counts[i];
    }

    // stride 
    params.cd_bs_stride = code_book.stride(0);
    params.cd_head_stride = code_book.stride(1);
    params.cd_b_stride = code_book.stride(2);
    params.sci_bs_stride = sorted_cluster_ids.stride(0);
    params.sci_head_stride = sorted_cluster_ids.stride(1);
    params.sci_b_stride = sorted_cluster_ids.stride(2);
    params.ckc_bs_stride = cluster_key_counts.stride(0);
    params.ckc_head_stride = cluster_key_counts.stride(1);
    params.ckc_b_stride = cluster_key_counts.stride(2);
    params.batchsize = batchsize;
    params.kv_heads = kv_heads;
    params.sub_spaces = sub_spaces;
    params.kv_len = kv_len;

    TORCH_CHECK(code_book.stride(3) == 1, "codebook.size(3) must be 1");
    TORCH_CHECK(sorted_cluster_ids.stride(3) == 1, "sorted_cluster_ids.size(3) must be 1");
    TORCH_CHECK(cluster_key_counts.stride(3) == 1, "cluster_key_counts.size(3) must be 1");

    if (num_clusters == 32) {
        convert_to_bits<32, TIER_NUM><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<32>) * TIER_NUM, stream>>>(params);
    } else if (num_clusters == 64) {
        convert_to_bits<64, TIER_NUM><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<64>) * TIER_NUM, stream>>>(params);
    } else if (num_clusters == 128) {
        convert_to_bits<128, TIER_NUM><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<128>) * TIER_NUM, stream>>>(params);
    } else if (num_clusters == 256) {
        convert_to_bits<256, TIER_NUM><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<256>) * TIER_NUM, stream>>>(params);
    } else if (num_clusters == 512) {
        convert_to_bits<512, TIER_NUM><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<512>) * TIER_NUM, stream>>>(params);
    } else if (num_clusters == 1024) {
        convert_to_bits<1024, TIER_NUM><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<1024>) * TIER_NUM, stream>>>(params);
    } else {
        TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");   \
    }

    // Grid X = batch*heads; Y chunks kv_len by BLOCK_SIZE.
    constexpr int BLOCK_SIZE = 1024;
    const unsigned int kBlockNumX = batchsize * kv_heads;
    const unsigned int kBlockNumY = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const auto grid = dim3{kBlockNumX, kBlockNumY};
    const auto block = dim3{256};

    int smem_size = (sub_spaces * TIER_NUM * num_clusters / 32) * sizeof(uint32_t);
    if (sub_spaces == 1) {
        if (num_clusters == 256) {
            launch_update_cache_cnt<1, 256, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 256");
        }
    } else if (sub_spaces == 8) {
        if (num_clusters == 256) {
            launch_update_cache_cnt<8, 256, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 256");
        }
    } else if (sub_spaces == 16) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<16, 32, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<16, 64, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<16, 128, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<16, 256, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
        }
    } else if (sub_spaces == 32) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<32, 32, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<32, 64, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<32, 128, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<32, 256, BLOCK_SIZE, TIER_NUM>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
        }
    } else{
        TORCH_CHECK(false, "sub_spaces must be 8, 16 or 32.");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));

    // return {tier_bits_tensor};
}








void update_cache_cnt(
    at::Tensor& sorted_cluster_ids,
    at::Tensor& cluster_key_counts,
    at::Tensor& code_book,
    const std::vector<uint32_t>& tier_counts,
    at::Tensor& cache_cnt
){
    // CHECK
    const int tier_num = tier_counts.size();
    if (tier_num == 1){
        update_cache_cnt_tier_xx<1>(sorted_cluster_ids, cluster_key_counts, code_book, tier_counts, cache_cnt);
    } else if (tier_num == 6){
        update_cache_cnt_tier_xx<6>(sorted_cluster_ids, cluster_key_counts, code_book, tier_counts, cache_cnt);
    } else {
        TORCH_CHECK(false, "tier_num must be 6");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_cache_cnt", &update_cache_cnt);
}