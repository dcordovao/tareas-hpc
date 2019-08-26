#include <iostream>

using namespace std;

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)

__global__
void add_vecs(int n, float *x, float *y, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<n)
  {
    z[i] = x[i]+y[i];
  }
}

int main(void)
{
  int N = 10;
  float *x, *y, *z, *d_x, *d_y, *d_z;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  z = (float*)malloc(N*sizeof(float));

  CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float))); // 1D array representation for grid 2D
  CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, N*sizeof(float)));
  
  for (int i=0; i<N; i++)
  {
    x[i] = i+1;
    y[i] = (i+1)*10; 
  }
  
  CUDA_CHECK(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice));

  int blockSize = 256; // # threads
  int gridSize = (N/blockSize)+1; // # blocks

  add_vecs<<<gridSize,blockSize>>>(N, d_x, d_y, d_z);
  cudaDeviceSynchronize();

  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  CUDA_CHECK(cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++)
  {
    cout << z[i] << endl;
  }
}