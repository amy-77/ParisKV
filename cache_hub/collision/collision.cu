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

struct CounterParams {
    const int *__restrict__ sorted_cluster_ids_ptr; // [bs, kv_heads, B, num_clusters]
    const int *__restrict__ topk_cluster_cnt_ptr;   // [bs, kv_heads, B]
    const int *__restrict__ code_book_ptr;   // [bs, kv_heads, kv_len, B]
    int *__restrict__ cache_cnt_ptr; // [bs, kv_heads, kv_len]
    uint32_t *__restrict__ cluster_ids_bits; // [bs, kv_heads, B, num_clusters/8]
    int batchsize;
    int kv_heads;
    int sub_spaces;
    int num_clusters;
    int kv_lens;
};


template<int Bits>
struct __align__(4) uint_bits{
    static_assert(Bits > 0, "Bit width must be positive.");
    static constexpr int StorageWords = (Bits + 31) / 32;
    uint32_t data[StorageWords];

    // __host__ __device__ __forceinline__
    // void set_pos(const int pos) {
    //     atomicOr( &data[pos / 32], 1u << (pos % 32));
    // }

    // __host__ __device__ __forceinline__
    // bool check_pos(const int pos) const  {
    //     return data[pos / 32] & (1u << (pos % 32));
    // }
};

template<int NUM_CLUSTER>
__global__ void convert_to_bits(const CounterParams params){

    const int warp_idx = threadIdx.x / 32;
    const int warp_num = blockDim.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    const int bidx = blockIdx.x;
    // input
    const int* cluster_ids_ptr = params.sorted_cluster_ids_ptr + bidx * NUM_CLUSTER;
    const int* cluster_end_ptr = params.topk_cluster_cnt_ptr + bidx;
    // output
    uint32_t* cluster_ids_bits_ptr = params.cluster_ids_bits + bidx * NUM_CLUSTER / 32;
    int tidx = threadIdx.x;

    const int end = __ldg(cluster_end_ptr);

    __shared__ uint_bits<NUM_CLUSTER> bits;

    if (tidx < NUM_CLUSTER / 32){
        bits.data[tidx] = 0u;
    }
    __syncthreads();

    int index = __ldg(cluster_ids_ptr + tidx);

    if (tidx < end){
        // bits.set_pos(index);
        atomicOr( &(bits.data[index / 32]), 1u << (index % 32));
    }

    __syncthreads();

    #pragma unroll
    for(int i = tidx; i < NUM_CLUSTER / 32; i += blockDim.x){
        cluster_ids_bits_ptr[i] = bits.data[i];
    }
}

template<int SUB_SPACE, int NUM_CLUSTERS, int BLOCK_SIZE>
__global__
void update_cache_cnt_kernel(const CounterParams params){
    const int warp_idx = threadIdx.x / 32;
    const int warp_num = blockDim.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int kv_lens = params.kv_lens;
    const int batchsize = params.batchsize;
    const int kv_heads = params.kv_heads;
    // const int* cluster_ids_ptr = params.sorted_cluster_ids_ptr + bidx * NUM_CLUSTERS * SUB_SPACE;
    const void* cluster_ids_ptr_bits = params.cluster_ids_bits + bidx * SUB_SPACE * NUM_CLUSTERS / 32;
    int* cache_cnt_ptr = params.cache_cnt_ptr + bidx * kv_lens;

    const int* code_book_ptr = params.code_book_ptr + bidx * SUB_SPACE * kv_lens;
    int tidx = threadIdx.x;

    // clear shared memory
    __shared__ uint_bits<NUM_CLUSTERS> cluster_ids_bits_smem[SUB_SPACE];

    // int4 (128-bit) loads
    #pragma unroll
    for (int i = tidx; i < SUB_SPACE * NUM_CLUSTERS / 32 / 4; i += blockDim.x) {

        const int4* gptr = reinterpret_cast<const int4*>(cluster_ids_ptr_bits) + i;
        int4 val = __ldg(gptr);

        int4* sptr = reinterpret_cast<int4*>(cluster_ids_bits_smem) + i;
        *sptr = val;
    }
    __syncthreads();

    if (tidx == 0 && bidx == 0 && bidy == 0){
    //     for(int i = 0 ; i < SUB_SPACE; i += 1) {
    //         printf("%d %d %d %d %d %d %d %d\n", cluster_ids_bits_smem[i].data[0], cluster_ids_bits_smem[i].data[1], 
    //                         cluster_ids_bits_smem[i].data[2], cluster_ids_bits_smem[i].data[3],
    //                         cluster_ids_bits_smem[i].data[4], cluster_ids_bits_smem[i].data[5], 
    //                         cluster_ids_bits_smem[i].data[6], cluster_ids_bits_smem[i].data[7]);
    //     }
    }
    const int start_k = bidy * BLOCK_SIZE;
    const int end_k = min(kv_lens, start_k + BLOCK_SIZE);


    #pragma unroll
    for (int ks = tidx + start_k; ks < end_k; ks += blockDim.x) {
        int cnt = 0;
        #pragma unroll
        for(int i = 0; i < SUB_SPACE; i += 1) {
            const int target_cluster_id = __ldg(code_book_ptr + ks * SUB_SPACE + i);
            bool hit = cluster_ids_bits_smem[i].data[target_cluster_id / 32] & (1u << (target_cluster_id % 32));
            if (hit) cnt += 1;
        }
        __syncwarp();
        cache_cnt_ptr[ks] = cnt;
    }
}

template<int SUB_SPACES, int NUM_CLUSTERS, int BLOCK_SIZE>
inline void launch_update_cache_cnt(dim3 grid, dim3 block, int smem_size,
                             const CounterParams &params) 
{

    cudaFuncSetAttribute(
        update_cache_cnt_kernel<SUB_SPACES, NUM_CLUSTERS, BLOCK_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    // launch
    update_cache_cnt_kernel<SUB_SPACES, NUM_CLUSTERS, BLOCK_SIZE><<<grid, block, smem_size>>>(params);
}


void update_cache_cnt_interface(
    at::Tensor sorted_cluster_ids,
    at::Tensor topk_cluster_cnt,
    at::Tensor code_book,
    at::Tensor cache_cnt
){
    /*
     * sorted_cluster_ids: [bs, kv_heads, B, num_clusters]
     * topk_cluster_cnt: [bs, kv_heads, B] clusters selected per sub-space
     * code_book: [bs, kv_heads, kv_len, B] cluster id per key per sub-space
     * cache_cnt: [bs, kv_heads, kv_len] collision count per key (sum over sub-spaces)
     */

    // check cuda
    CHECK_DEVICE(sorted_cluster_ids);
    CHECK_DEVICE(topk_cluster_cnt);
    CHECK_DEVICE(code_book);

    // check continguous
    CHECK_CONTIGUOUS(sorted_cluster_ids);
    CHECK_CONTIGUOUS(topk_cluster_cnt);
    CHECK_CONTIGUOUS(code_book);

    // check shape
    TORCH_CHECK(sorted_cluster_ids.dim() == 4);
    TORCH_CHECK(topk_cluster_cnt.dim() == 3);
    TORCH_CHECK(code_book.dim() == 4);
    const int batchsize = sorted_cluster_ids.size(0);
    const int kv_heads = sorted_cluster_ids.size(1);
    const int sub_spaces = sorted_cluster_ids.size(2);
    const int num_clusters = sorted_cluster_ids.size(3);
    const int kv_lens = code_book.size(2);
    TORCH_CHECK(topk_cluster_cnt.size(0) == batchsize);
    TORCH_CHECK(code_book.size(0) == batchsize);
    TORCH_CHECK(topk_cluster_cnt.size(1) == kv_heads);
    TORCH_CHECK(code_book.size(1) == kv_heads);
    TORCH_CHECK(topk_cluster_cnt.size(2) == sub_spaces);
    TORCH_CHECK(code_book.size(3) == sub_spaces);
    TORCH_CHECK(num_clusters % 32 == 0, "num_clusters must be a multiple of 32");
    TORCH_CHECK(sub_spaces % 8 == 0, "sub_spaces must be a multiple of 8");

    // Grid X = batch*heads; Y chunks kv_len by BLOCK_SIZE.
    constexpr int BLOCK_SIZE = 1024;
    const unsigned int kBlockNumX = batchsize * kv_heads;
    const unsigned int kBlockNumY = (kv_lens + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // launch kernel
    const c10::Device dev = cache_cnt.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid = dim3{kBlockNumX, kBlockNumY};
    // const auto block = dim3{min(kThreadsPerBlock, sub_spaces * 32)};
    const auto block = dim3{256};
    
    torch::Tensor cluster_ids_bits = torch::zeros({batchsize, kv_heads, sub_spaces, num_clusters / 32}, sorted_cluster_ids.options().dtype(torch::kUInt32));

    CounterParams params{
        sorted_cluster_ids.data_ptr<int>(),
        topk_cluster_cnt.data_ptr<int>(),
        code_book.data_ptr<int>(),
        cache_cnt.data_ptr<int>(),
        cluster_ids_bits.data_ptr<uint32_t>(),
        batchsize,
        kv_heads,
        sub_spaces,
        num_clusters,
        kv_lens
    };

    if (num_clusters == 32) {
        convert_to_bits<32><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<32>) >>>(params);
    } else if (num_clusters == 64) {    
        convert_to_bits<64><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<64>) >>>(params);
    } else if (num_clusters == 128) {
        convert_to_bits<128><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<128>) >>>(params);
    } else if (num_clusters == 256) {
        convert_to_bits<256><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<256>) >>>(params);
    } else {
        TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
    }
    // std::cout << sorted_cluster_ids[0][0][0] << std::endl;
    // std::cout << topk_cluster_cnt[0][0][0] << std::endl;
    // std::cout << cluster_ids_bits[0][0][0] << std::endl;

    int smem_size = (sub_spaces * num_clusters + BLOCK_SIZE) * 4;
    if (sub_spaces == 16) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<16, 32, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<16, 64, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<16, 128, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<16, 256, BLOCK_SIZE>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
        }
    } else if (sub_spaces == 32) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<32, 32, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<32, 64, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<32, 128, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<32, 256, BLOCK_SIZE>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
        }
    } else{
        TORCH_CHECK(false, "sub_spaces must be 8, 16 or 32.");
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_cache_cnt", &update_cache_cnt_interface);
}