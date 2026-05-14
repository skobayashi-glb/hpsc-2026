#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void count_kernel(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  for (int i=0; i<range; i++) bucket[i] = 0;

  count_kernel<<<1, n>>>(key, bucket, n);
  cudaDeviceSynchronize();

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
