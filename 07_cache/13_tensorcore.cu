#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
using namespace std;
using namespace nvcuda;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
             cudaGetErrorString(err));                                       \
      return 1;                                                              \
    }                                                                        \
  } while (0)

// 入力は元の float* のまま受け取り、shared memory に置く直前で half 化する。
__global__ __launch_bounds__(256)
void kernel(int dim_m, int dim_n, int dim_k,
	    const float *__restrict__ d_a,
	    const float *__restrict__ d_b,
	    float *__restrict__ d_c) {
  int offset_m = 128 * blockIdx.x;
  int offset_n = 256 * blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = threadIdx.x / 32;
  int warp_m_idx = warp_id / 4;
  int warp_n_idx = warp_id % 4;

  constexpr int SMEM_A_LD = 136;
  constexpr int SMEM_B_LD = 40;
  __shared__ __align__(16) half smem_a[32][SMEM_A_LD];
  __shared__ __align__(16) half smem_b[256][SMEM_B_LD];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += 32) {
    __syncthreads();
    for (int iter = 0; iter < 4; iter++) {
      int flat = iter * 256 + tid;
      int j = flat / 32;
      int m_local = (flat % 32) * 4;
      const float4 v =
          *reinterpret_cast<const float4 *>(
              &d_a[(k + j) * dim_m + offset_m + m_local]);
      const half2 h01 = __float22half2_rn(make_float2(v.x, v.y));
      const half2 h23 = __float22half2_rn(make_float2(v.z, v.w));
      *reinterpret_cast<half2 *>(&smem_a[j][m_local + 0]) = h01;
      *reinterpret_cast<half2 *>(&smem_a[j][m_local + 2]) = h23;
    }
    for (int iter = 0; iter < 8; iter++) {
      int flat    = iter * 256 + tid;
      int n_local = flat / 8;
      int k_local = (flat % 8) * 4;
      const float4 v =
          *reinterpret_cast<const float4 *>(
              &d_b[(offset_n + n_local) * dim_k + k + k_local]);
      const half2 h01 = __float22half2_rn(make_float2(v.x, v.y));
      const half2 h23 = __float22half2_rn(make_float2(v.z, v.w));
      *reinterpret_cast<half2 *>(&smem_b[n_local][k_local + 0]) = h01;
      *reinterpret_cast<half2 *>(&smem_b[n_local][k_local + 2]) = h23;
    }
    __syncthreads();
    for (int kk = 0; kk < 32; kk += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];
      for (int c = 0; c < 4; c++) {
        wmma::load_matrix_sync(b_frag[c], &smem_b[warp_n_idx * 64 + c * 16][kk], SMEM_B_LD);
      }
      for (int r = 0; r < 4; r++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &smem_a[kk][warp_m_idx * 64 + r * 16], SMEM_A_LD);
        for (int c = 0; c < 4; c++) {
          wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
        }
      }
    }
  }
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      int c_m = offset_m + warp_m_idx * 64 + r * 16;
      int c_n = offset_n + warp_n_idx * 64 + c * 16;
      if (c_m < dim_m && c_n < dim_n)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  printf("Kernel variant: WMMA 128x256 tile, 8 warps, float input, padded smem\n");
  float *A, *B;
  CUDA_CHECK(cudaMallocManaged(&A, (size_t)m * k * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&B, (size_t)k * n * sizeof(float)));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaMemLocation dev_loc{};
  dev_loc.type = cudaMemLocationTypeDevice;
  dev_loc.id = dev;
  CUDA_CHECK(cudaMemPrefetchAsync(A, (size_t)m * k * sizeof(float),
                                  dev_loc, 0));
  CUDA_CHECK(cudaMemPrefetchAsync(B, (size_t)k * n * sizeof(float),
                                  dev_loc, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  float *C, *C2;
  CUDA_CHECK(cudaMalloc((void **)&C, (size_t)m * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&C2, (size_t)m * n * sizeof(float)));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  dim3 block = dim3(256);
  dim3 grid = dim3((m+127)/128, (n+255)/256);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m, n, k, A, B, C2);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
      printf("kernel launch failed: %s\n", cudaGetErrorString(launch_status));
      return 1;
    }
    launch_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess) {
      printf("kernel execution failed: %s\n", cudaGetErrorString(launch_status));
      return 1;
    }
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS:    %.2f Gflops, %.6f sec\n", cublas_flops, tcublas);
  printf("MY_KERNEL: %.2f Gflops, %.6f sec\n", cutlass_flops, tcutlass);
  printf("Kernel GFLOPS: %.2f\n", cutlass_flops);
  printf("Ratio:     %.2f %% of cuBLAS\n", 100.0 * cutlass_flops / cublas_flops);
  float *C_host = (float *)malloc((size_t)m * n * sizeof(float));
  float *C2_host = (float *)malloc((size_t)m * n * sizeof(float));
  if (C_host == nullptr || C2_host == nullptr) {
    printf("host malloc failed\n");
    return 1;
  }
  CUDA_CHECK(cudaMemcpy(C_host, C, (size_t)m * n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(C2_host, C2, (size_t)m * n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C_host[m*i+j] - C2_host[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  free(C_host);
  free(C2_host);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
