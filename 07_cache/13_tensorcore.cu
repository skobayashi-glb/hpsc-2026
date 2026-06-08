#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t stat = (call);                                             \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                   __LINE__, int(stat));                                      \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void float_to_half(const float *__restrict__ src,
                              half *__restrict__ dst,
                              int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

__global__ __launch_bounds__(512, 1)
void kernel(int dim_m, int dim_n, int dim_k,
            const half *__restrict__ d_a,
            const half *__restrict__ d_b,
            float *__restrict__ d_c) {
  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 32;
  constexpr int BM_PAD = 136;
  constexpr int BK_PAD = 40;
  constexpr int WM = 32;
  constexpr int WN = 64;

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int warp_m = warp_id >> 2;
  int warp_n = warp_id & 3;
  int base_m = blockIdx.x * BM;
  int base_n = blockIdx.y * BN;

  // C の 1 block あたりの担当範囲を 128x256 に拡大。
  // [変更] 16 warp で 4x4 個の 32x64 warp tile を計算する。
  // 1 warp が持つ accumulator を 16 個から 8 個に減らし、レジスタ圧迫を下げる。
  // [変更] 通常ロードでは double buffer が十分に重ならないため、単一バッファに戻して
  // shared memory に padding を入れる。WMMA load の bank conflict を減らす狙い。
  __shared__ __align__(16) half smem_a[BK][BM_PAD];
  __shared__ __align__(16) half smem_b[BN][BK_PAD];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  for (int k0 = 0; k0 < dim_k; k0 += BK) {
    // A/B は計測前に half 化済みなので、カーネル内の float->half 変換をなくす。
    // [変更] 本番サイズは 128/256/32 のタイルで割り切れるため、境界分岐を外す。
    // さらに half2 で 2 要素ずつ shared memory に運び、ロード命令数を減らす。
    for (int idx = tid; idx < BK * (BM / 2); idx += blockDim.x) {
      int kk = idx / (BM / 2);
      int mm2 = idx - kk * (BM / 2);
      int mm = mm2 * 2;
      const half2 *src =
          reinterpret_cast<const half2 *>(&d_a[(k0 + kk) * dim_m + base_m + mm]);
      half2 *dst = reinterpret_cast<half2 *>(&smem_a[kk][mm]);
      *dst = *src;
    }
    for (int idx = tid; idx < BN * (BK / 2); idx += blockDim.x) {
      int nn = idx / (BK / 2);
      int kk2 = idx - nn * (BK / 2);
      int kk = kk2 * 2;
      const half2 *src =
          reinterpret_cast<const half2 *>(&d_b[(base_n + nn) * dim_k + k0 + kk]);
      half2 *dst = reinterpret_cast<half2 *>(&smem_b[nn][kk]);
      *dst = *src;
    }
    __syncthreads();

    // WMMA の固定回数ループを unroll し、Tensor Core 命令の制御オーバーヘッドを下げる。
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                       wmma::col_major>
            a_frag;
        int a_m = warp_m * WM + r * 16;
        wmma::load_matrix_sync(a_frag, &smem_a[kk][a_m], BM_PAD);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                         wmma::col_major>
              b_frag;
          int b_n = warp_n * WN + c * 16;
          wmma::load_matrix_sync(b_frag, &smem_b[b_n][kk], BK_PAD);
          wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      int c_m = base_m + warp_m * WM + r * 16;
      int c_n = base_n + warp_n * WN + c * 16;
      // 本番サイズは 16 の倍数なので store 側の境界分岐も外す。
      wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c],
                              dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
#ifdef DEBUG_SMALL
  int m = 256;
  int k = 256;
  int n = 256;
#else
  int m = 10240;
  int k = 4096;
  int n = 8192;
#endif
  float alpha = 1.0f;
  float beta = 0.0f;
  int Nt = 20;

  std::printf("Kernel variant: 128x256 tile, 16 warps, padded smem, half2 load\n");

  float *A, *B, *C, *C2;
  CUDA_CHECK(cudaMallocManaged(&A, int64_t(m) * k * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&B, int64_t(k) * n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C, int64_t(m) * n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C2, int64_t(m) * n * sizeof(float)));

  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      A[k * i + j] = drand48();
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      B[n * i + j] = drand48();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      C[m * i + j] = C2[m * i + j] = 0.0f;

  half *A_half, *B_half;
  CUDA_CHECK(cudaMallocManaged(&A_half, int64_t(m) * k * sizeof(half)));
  CUDA_CHECK(cudaMallocManaged(&B_half, int64_t(k) * n * sizeof(half)));
  // A/B を half に事前変換する。変換時間は GEMM の性能測定に含めない。
  float_to_half<<<(int64_t(m) * k + 255) / 256, 256>>>(A, A_half,
                                                        int64_t(m) * k);
  CUDA_CHECK(cudaGetLastError());
  float_to_half<<<(int64_t(k) * n + 255) / 256, 256>>>(B, B_half,
                                                        int64_t(k) * n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaMemPrefetchAsync(A_half, int64_t(m) * k * sizeof(half), dev));
  CUDA_CHECK(cudaMemPrefetchAsync(B_half, int64_t(k) * n * sizeof(half), dev));
  CUDA_CHECK(cudaMemPrefetchAsync(C, int64_t(m) * n * sizeof(float), dev));
  CUDA_CHECK(cudaMemPrefetchAsync(C2, int64_t(m) * n * sizeof(float), dev));
  CUDA_CHECK(cudaDeviceSynchronize());

  // shared memory を優先する carveout を指定し、48 KB の double-buffer tile を安定して使う。
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));

  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto run_cublas = [&]() {
    CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              A_half, CUDA_R_16F, m,
                              B_half, CUDA_R_16F, k,
                              &beta,
                              C, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  // cuBLAS も half 入力にそろえ、自作 Tensor Core カーネルと同じ条件で比較する。
  for (int i = 0; i < 2; i++) {
    run_cublas();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < Nt; i++) {
    run_cublas();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float cublas_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
  int64_t num_flops = 2 * int64_t(m) * int64_t(n) * int64_t(k) +
                      2 * int64_t(m) * int64_t(n);
  double tcublas = double(cublas_ms) / 1000.0 / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block(512);
  dim3 grid((m + 127) / 128, (n + 255) / 256);
  // chrono ではなく CUDA Event で、GPU 上のカーネル実行時間だけを測る。
  for (int i = 0; i < 2; i++) {
    kernel<<<grid, block>>>(m, n, k, A_half, B_half, C2);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < Nt; i++) {
    kernel<<<grid, block>>>(m, n, k, A_half, B_half, C2);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  double tkernel = double(kernel_ms) / 1000.0 / Nt;
  double kernel_flops = double(num_flops) / tkernel / 1.0e9;
  std::printf("CUBLAS:    %.2f Gflops, %.6f sec\n",
              cublas_flops, tcublas);
  std::printf("MY_KERNEL: %.2f Gflops, %.6f sec\n",
              kernel_flops, tkernel);
  std::printf("Ratio:     %.2f %% of cuBLAS\n",
              100.0 * kernel_flops / cublas_flops);

  CUDA_CHECK(cudaMemPrefetchAsync(C, int64_t(m) * n * sizeof(float),
                                  cudaCpuDeviceId));
  CUDA_CHECK(cudaMemPrefetchAsync(C2, int64_t(m) * n * sizeof(float),
                                  cudaCpuDeviceId));
  CUDA_CHECK(cudaDeviceSynchronize());

  double err_sum = 0.0;
  double err_max = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      double err = std::fabs(C[m * i + j] - C2[m * i + j]);
      err_sum += err;
      if (err > err_max) err_max = err;
    }
  }
  std::printf("mean error: %e\n", err_sum / n / m);
  std::printf("max error:  %e\n", err_max);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(C2));
  CUDA_CHECK(cudaFree(A_half));
  CUDA_CHECK(cudaFree(B_half));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
