#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  int offset_m = 128 * blockIdx.x;
  int offset_n = 128 * blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = threadIdx.x / 32;
  int warp_m_idx = warp_id / 2;
  int warp_n_idx = warp_id % 2;

  __shared__ half smem_a[32][128];  // [K][M]
  // [変更] smem_b を [K][N] → [N][K] に転置
  // 理由: Bのグローバルメモリロードをcoalesced（連続アドレス）にするため
  __shared__ half smem_b[128][32];  // [N][K]

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += 32) {
    __syncthreads();
    // A: coalescedロード（変更なし）
    // 連続スレッドが連続アドレスを読む ✓
    for (int j = 0; j < 32; j++)
      smem_a[j][tid] = __float2half(d_a[(k + j) * dim_m + offset_m + tid]);

    // [変更] B: coalescedロード（フラットインデックス方式）
    // flat = iter*128 + tid を (n_local, k_local) に分解する
    // warp内の連続スレッドが同じ行(n_local)の連続K列を読む → coalesced ✓
    for (int iter = 0; iter < 32; iter++) {
      int flat    = iter * 128 + tid;
      int n_local = flat / 32;   // 担当するN方向インデックス
      int k_local = flat % 32;   // 担当するK方向インデックス
      smem_b[n_local][k_local] = __float2half(d_b[(offset_n + n_local) * dim_k + k + k_local]);
    }
    __syncthreads();

    for (int kk = 0; kk < 32; kk += 16) {
      for (int r = 0; r < 4; r++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &smem_a[kk][warp_m_idx * 64 + r * 16], 128);
        for (int c = 0; c < 4; c++) {
          // [変更] smem_b が [N][K] になったので col_major に変更、stride 128→32
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, &smem_b[warp_n_idx * 64 + c * 16][kk], 32);
          wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
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
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
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
  int tile = 128;
  dim3 block = dim3(128);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
