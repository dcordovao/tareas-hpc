#include <stdio.h>
#include <iostream>
#include <iostream>
#include <fstream>
#include <random>


#define WIDTH 8192
#define LENGHT 8192
#define N_PARTICLES 5000
#define INF 999999.999
#define RADIO 100
#define CELLS_FOR_THREAD 8

using namespace std;

// __constant__ float x_part_dev[N_PARTICLES];
// __constant__ float y_part_dev[N_PARTICLES];

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)

__global__
void minReduction(float *in, float *out)
{
  __shared__ float sharedData[256];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + tid; // blockSize = 256
  sharedData[tid] = in[i] + in[i+blockDim.x];
  __syncthreads();

  for (unsigned int  s = blockDim.x/2; s>32; s>>=1) {
    if(tid<s)
    {
      sharedData[tid] = (sharedData[tid]<sharedData[tid+s])?sharedData[tid]:sharedData[tid+s];
    }
    __syncthreads();
  }

  if (tid < 32)
  {
    sharedData[tid] = (sharedData[tid]<sharedData[tid+32])?sharedData[tid]:sharedData[tid+32];
    sharedData[tid] = (sharedData[tid]<sharedData[tid+16])?sharedData[tid]:sharedData[tid+16];
    sharedData[tid] = (sharedData[tid]<sharedData[tid+8])?sharedData[tid]:sharedData[tid+8];
    sharedData[tid] = (sharedData[tid]<sharedData[tid+4])?sharedData[tid]:sharedData[tid+4];
    sharedData[tid] = (sharedData[tid]<sharedData[tid+2])?sharedData[tid]:sharedData[tid+2];
    sharedData[tid] = (sharedData[tid]<sharedData[tid+1])?sharedData[tid]:sharedData[tid+1];
  }

  if(tid==0)
  {
    out[blockIdx.x] = sharedData[0];
  }
}

float random_float(float min, float max) {

	return ((float)rand() / RAND_MAX) * (max - min) + min;

}


int main(int argc, char *argv[]){
  // Load data
  string input_file_name;

  /*
  if (argc > 1) {
		input_file_name = argv[1];
	} else {
		cout << "faltó un argumento" << endl;
		exit(0);
	}

	ifstream infile;
	cout << input_file_name.c_str() << endl;
    infile.open(input_file_name.c_str());
    */
  int nP = WIDTH*LENGHT;

  float *cells;

  // infile >> nP;
  // cout << "nP: "<<nP << endl;

  cells = (float *)malloc(WIDTH*LENGHT * sizeof(float));
  //cells = (float*)malloc(nP*sizeof(float));


  int target_min_pos = 1000000; 
  for (int i = 0; i<nP; i++){
    if (i != target_min_pos){
    cells[i] = 1.0f;
    } else {
    cells[i] = random_float(2.0f, 19.99f);
    }
  }

  // Get memory for structures
  float *chunk,*outData,*out2,y[128];

  // Define sizes of GPU
  int blockSize = 256; // # threads
  int gridSize = ((WIDTH*LENGHT)/256)/CELLS_FOR_THREAD; // # blocks

  cout << "gridSize: " << gridSize << endl;
  // Get memory in GPU for structures

  // data for charge function
  CUDA_CHECK(cudaMalloc(&chunk, gridSize*sizeof(float))); // 1D array representation for grid 2D

  // data for reduction function
  CUDA_CHECK(cudaMalloc(&outData, gridSize*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out2, (gridSize/blockSize)*sizeof(float)));

  float min = INF;
  float *aux;
  aux = (float*)malloc(gridSize*sizeof(float));

  // Search min load
  for (size_t i = 0; i < CELLS_FOR_THREAD; i++) {
    memcpy(aux, cells + i*gridSize, gridSize * sizeof(float));

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(chunk, aux, gridSize*sizeof(float), cudaMemcpyHostToDevice));
    minReduction<<<gridSize,blockSize>>>(chunk,outData); // outData lenght 32.768
    cudaDeviceSynchronize();
    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    minReduction<<<gridSize/blockSize,blockSize>>>(outData,out2); // out2 lenght 128
    cudaDeviceSynchronize();
    cudaMemcpy(y, out2, 128*sizeof(float), cudaMemcpyDeviceToHost); 
    // check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    // min load
    for (size_t i = 0; i < 128; i++) {
      min = (y[i]<min)?y[i]:min;
    }
    cout << min << endl;
  }

  cudaFree(chunk);
  cudaFree(outData);
  cudaFree(out2);
  free(cells);
  free(aux);

  return 0;
}
