#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>

#define N 10000
#define MIN_POS 1688

using namespace std;

typedef struct
{
    float charge;
    int index;
} cell;

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)

__global__ void reduce0(cell *g_idata, cell *g_odata) 
{ 
    extern __shared__ cell sdata[];
    // each thread loads one element from global to shared mem 
    unsigned int tid = threadIdx.x; 
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem 
    for(unsigned int s=1; s < blockDim.x; s *= 2) 
    { 
        if (tid % (2*s) == 0) 
        { 
            sdata[tid] = (sdata[tid].charge < sdata[tid + s].charge)? sdata[tid]: sdata[tid + s]; 
        } 
        __syncthreads(); 
    }
    // write result for this block to global mem 
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


int main()
{
    cell *a, *out, *dev_a, *dev_out;
    
    a = (cell*)malloc(N*sizeof(cell));
    out = (cell*)malloc(N*sizeof(cell));
    CUDA_CHECK(cudaMalloc(&dev_a, N*sizeof(cell)));
    CUDA_CHECK(cudaMalloc(&dev_out, N*sizeof(cell)));

    for (int i=0; i<N; i++)
    {
        if (i==MIN_POS) a[i].charge = 1;
        else a[i].charge = (i%15)+50;

        a[i].index = i;
    }

    CUDA_CHECK(cudaMemcpy(dev_a, a, N*sizeof(cell), cudaMemcpyHostToDevice));

    int blockSize = 256; // # threads
    int gridSize = N/blockSize + (N % blockSize != 0);; // # blocks
    int sharedBytes = blockSize*sizeof(cell); 

    reduce0<<<gridSize,blockSize,sharedBytes>>>(dev_a, dev_out);
    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    CUDA_CHECK(cudaMemcpy(out, dev_out, N*sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Min charge:" << out[0].charge << endl;
    cout << "Min index:" << out[0].index << endl;
}
