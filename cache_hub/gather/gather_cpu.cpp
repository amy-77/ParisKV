#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <cstring>
#include <iostream>

/*
  CPU gather specialized for KV layout (Optimized version):
    src:   [B, H, N, D]  (CPU)
    index: [B, H, K]     (CPU, int64)
    out:   [B, H, K, D]  (CPU, preallocated)

  Optimizations:
    1. Eliminate division in hot loop (use nested loops)
    2. Software prefetch for better cache utilization
    3. Use memcpy for contiguous memory (compiler optimizes well)
    4. __restrict__ hints for compiler optimization
    5. Bounds check moved outside hot path
*/

template <typename scalar_t>
static void gather_kv_out_cpu_impl(
    const scalar_t* __restrict__ src_ptr,
    const int64_t* __restrict__ idx_ptr,
    scalar_t* __restrict__ out_ptr,
    int64_t B, int64_t H, int64_t N, int64_t D, int64_t K,
    int64_t src_s0, int64_t src_s1, int64_t src_s2, int64_t src_s3,
    int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3) {

  const int64_t total_bh = B * H;
  const size_t row_bytes = D * sizeof(scalar_t);
  const bool is_contiguous = (src_s3 == 1 && out_s3 == 1);

  // Prefetch distance (tunable, 4-8 works well for most cases)
  constexpr int64_t PREFETCH_DIST = 4;

  // Debug prints disabled
  // static bool printed = false;
  // if (!printed) {
  //   std::cout << "intra-op threads: " << at::get_num_threads() << "\n";
  //   std::cout << "inter-op threads: " << at::get_num_interop_threads() << "\n";
  //   std::cout << "[gather_kv_out_cpu_impl] B=" << B << " H=" << H << " K=" << K
  //             << " D=" << D << " total_bh=" << total_bh
  //             << " contiguous=" << is_contiguous
  //             << std::endl;
  //   printed = true;
  // }

  // Parallel over (batch, head) pairs - avoids division in hot loop
  at::parallel_for(0, total_bh, 1, [&](int64_t bh_begin, int64_t bh_end) {
    using Vec = at::vec::Vectorized<scalar_t>;

    for (int64_t bh = bh_begin; bh < bh_end; ++bh) {
      // Compute b, h only once per (batch, head) pair
      const int64_t b = bh / H;
      const int64_t h = bh - b * H;  // Faster than modulo
      
      const int64_t* idx_base = idx_ptr + bh * K;
      const scalar_t* src_base = src_ptr + b * src_s0 + h * src_s1;
      scalar_t* out_base = out_ptr + b * out_s0 + h * out_s1;
      
      // Prefetch first few rows
      for (int64_t p = 0; p < std::min(PREFETCH_DIST, K); ++p) {
        const int64_t idx = idx_base[p];
        __builtin_prefetch(src_base + idx * src_s2, 0, 0);
      }
      
      if (is_contiguous) {
        // Fast path: contiguous memory, use memcpy
        for (int64_t k = 0; k < K; ++k) {
          // Prefetch ahead
          if (k + PREFETCH_DIST < K) {
            const int64_t next_idx = idx_base[k + PREFETCH_DIST];
            __builtin_prefetch(src_base + next_idx * src_s2, 0, 0);
          }
          
          const int64_t idx = idx_base[k];
          #ifndef NDEBUG
      TORCH_CHECK(idx >= 0 && idx < N,
                  "index ", idx, " is out of bounds for dimension 2 with size ", N);
          #endif

          const scalar_t* src_row = src_base + idx * src_s2;
          scalar_t* out_row = out_base + k * out_s2;

          // memcpy is highly optimized by compiler (often uses AVX)
          std::memcpy(out_row, src_row, row_bytes);
        }
      } else {
        // Slow path: non-contiguous, use SIMD or element-wise
        for (int64_t k = 0; k < K; ++k) {
          if (k + PREFETCH_DIST < K) {
            const int64_t next_idx = idx_base[k + PREFETCH_DIST];
            __builtin_prefetch(src_base + next_idx * src_s2, 0, 0);
          }
          
          const int64_t idx = idx_base[k];
          #ifndef NDEBUG
          TORCH_CHECK(idx >= 0 && idx < N,
                      "index ", idx, " is out of bounds for dimension 2 with size ", N);
          #endif
          
          const scalar_t* src_row = src_base + idx * src_s2;
          scalar_t* out_row = out_base + k * out_s2;
          
          // Element-wise copy for non-contiguous
        for (int64_t d = 0; d < D; ++d) {
          out_row[d * out_s3] = src_row[d * src_s3];
          }
        }
      }
    }
  });
}

static void gather_kv_out_cpu(torch::Tensor src, torch::Tensor index, torch::Tensor out) {
  TORCH_CHECK(src.device().is_cpu(), "src must be on CPU");
  TORCH_CHECK(index.device().is_cpu(), "index must be on CPU");
  TORCH_CHECK(out.device().is_cpu(), "out must be on CPU");

  TORCH_CHECK(src.dim() == 4, "src must be 4D [B,H,N,D]");
  TORCH_CHECK(index.dim() == 3, "index must be 3D [B,H,K]");
  TORCH_CHECK(out.dim() == 4, "out must be 4D [B,H,K,D]");

  TORCH_CHECK(index.scalar_type() == torch::kInt64, "index must be int64");
  TORCH_CHECK(src.scalar_type() == out.scalar_type(), "src/out dtype must match");

  const auto B = src.size(0);
  const auto H = src.size(1);
  const auto N = src.size(2);
  const auto D = src.size(3);
  const auto K = index.size(2);

  TORCH_CHECK(index.size(0) == B && index.size(1) == H, "index shape mismatch");
  TORCH_CHECK(out.size(0) == B && out.size(1) == H && out.size(2) == K && out.size(3) == D,
              "out shape mismatch");

  // Bounds check all indices upfront (outside hot path)
  #ifdef NDEBUG
  {
    const int64_t* idx_ptr = index.data_ptr<int64_t>();
    const int64_t total_idx = B * H * K;
    for (int64_t i = 0; i < total_idx; ++i) {
      TORCH_CHECK(idx_ptr[i] >= 0 && idx_ptr[i] < N,
                  "index ", idx_ptr[i], " is out of bounds for dimension 2 with size ", N);
    }
  }
  #endif

  // Strides are in elements
  const auto src_s0 = src.stride(0);
  const auto src_s1 = src.stride(1);
  const auto src_s2 = src.stride(2);
  const auto src_s3 = src.stride(3);

  const auto out_s0 = out.stride(0);
  const auto out_s1 = out.stride(1);
  const auto out_s2 = out.stride(2);
  const auto out_s3 = out.stride(3);

  AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, src.scalar_type(), "gather_kv_out_cpu", [&] {
    gather_kv_out_cpu_impl<scalar_t>(
        src.data_ptr<scalar_t>(),
        index.data_ptr<int64_t>(),
        out.data_ptr<scalar_t>(),
        B, H, N, D, K,
        src_s0, src_s1, src_s2, src_s3,
        out_s0, out_s1, out_s2, out_s3);
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_kv_out_cpu", &gather_kv_out_cpu, "KV gather (CPU, out=, optimized with prefetch)");
}
