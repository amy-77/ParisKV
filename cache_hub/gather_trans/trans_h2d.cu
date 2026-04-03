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


template<int DIM>
__global__ 
void h2d_gather_kv_kernel(
    void* __restrict__ src_k_caches_ptr,
    void* __restrict__ src_v_caches_ptr,
    void* __restrict__ dst_k_caches_ptr,
    void* __restrict__ dst_v_caches_ptr,
    void* __restrict__ selected_indices_ptr,
    int src_len,
    int dst_len
) {
    const int bs = gridDim.x;
    const int heads = gridDim.y;
    const int topk = gridDim.z;
    const int tidx = threadIdx.x;

    constexpr int uint4_per_dim = DIM * 2 / 16; // 256 / 16 = 16
    if (DIM == 128) {
        // Calculate the index for this thread - use int64_t for indices
        int64_t index = __ldg(reinterpret_cast<int64_t*>(selected_indices_ptr)
                             + blockIdx.x * heads * topk + blockIdx.y * topk + blockIdx.z);
        
        // Process k-cache
        if (tidx < 16) {
            uint4* src_ptr_k = reinterpret_cast<uint4*>(src_k_caches_ptr);
            uint4* dst_ptr_k = reinterpret_cast<uint4*>(dst_k_caches_ptr);
            
            // Source: read from the index position in source tensor
            src_ptr_k += (blockIdx.x * heads * src_len * uint4_per_dim) 
                         + (blockIdx.y * src_len * uint4_per_dim)
                         + (index * uint4_per_dim);
            
            // Destination: write to the blockIdx.z position in dest tensor (topk index)
            dst_ptr_k += (blockIdx.x * heads * dst_len * uint4_per_dim) 
                         + (blockIdx.y * dst_len * uint4_per_dim)
                         + (blockIdx.z * uint4_per_dim);

            dst_ptr_k[tidx] = src_ptr_k[tidx];
        }
        // Process v-cache  
        else if (tidx < 32) {
            uint4* src_ptr_v = reinterpret_cast<uint4*>(src_v_caches_ptr);
            uint4* dst_ptr_v = reinterpret_cast<uint4*>(dst_v_caches_ptr);
            int v_tidx = tidx - 16;
            
            // Source: read from the index position in source tensor  
            src_ptr_v += (blockIdx.x * heads * src_len * uint4_per_dim) 
                         + (blockIdx.y * src_len * uint4_per_dim)
                         + (index * uint4_per_dim);
            
            // Destination: write to the blockIdx.z position in dest tensor (topk index)
            dst_ptr_v += (blockIdx.x * heads * dst_len * uint4_per_dim) 
                         + (blockIdx.y * dst_len * uint4_per_dim)
                         + (blockIdx.z * uint4_per_dim);

            dst_ptr_v[v_tidx] = src_ptr_v[v_tidx];
        }
    }
}

void h2d_gather_kv(
    at::Tensor& src_k_caches, // bs, heads, kv_len, dim
    at::Tensor& src_v_caches, // bs, heads, kv_len, dim
    at::Tensor& selected_indices, // bs, heads, topk
    at::Tensor& dst_k_caches, // bs, heads, max_topk, dim
    at::Tensor& dst_v_caches // bs, heads, max_topk, dim
){
    void * src_k_caches_ptr = src_k_caches.data_ptr();
    void * src_v_caches_ptr = src_v_caches.data_ptr();
    void * dst_k_caches_ptr = dst_k_caches.data_ptr();
    void * dst_v_caches_ptr = dst_v_caches.data_ptr();
    void * selected_indices_ptr = selected_indices.data_ptr();

    unsigned int bs = src_k_caches.size(0);
    unsigned int heads = src_k_caches.size(1);
    unsigned int src_len = src_k_caches.size(2);
    unsigned int dim = src_k_caches.size(3);
    unsigned int topk = selected_indices.size(2);
    unsigned int dst_len = dst_k_caches.size(2);

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const c10::Device dev = dst_v_caches.device();
    c10::cuda::CUDAGuard device_guard(dev);

    dim3 grid = {bs, heads, topk};
    if (dim == 128) {
        h2d_gather_kv_kernel<128><<<grid, 32, 0, stream>>>(
            src_k_caches_ptr, 
            src_v_caches_ptr,
            dst_k_caches_ptr,
            dst_v_caches_ptr,
            selected_indices_ptr,
            src_len, dst_len
        );
    } else {
        TORCH_CHECK(false, "The dim must be 128.");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h2d_gather_kv", &h2d_gather_kv);
}