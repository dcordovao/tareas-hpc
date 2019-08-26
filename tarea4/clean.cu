#include <stdio.h>
#include <iostream>
#include <iostream>
#include <fstream>

#define WIDTH 8192
#define LENGHT 8192
#define N_PARTICLES 5000
#define RADIO 100

using namespace std;

typedef struct
{
    float charge;
    int index;
} cell;

__constant__ float x_part_dev[N_PARTICLES];
__constant__ float y_part_dev[N_PARTICLES];

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)


__device__ float dist(float x1, float y1, float x2, float y2)
{
  float dist;
  dist = sqrtf(powf(x2-x1, 2) + powf(y2-y1, 2));
  if(dist != 0) return 1/dist;
  else return -1;
}



__global__ void charge(float l, cell *map)
{

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float rowParticle,colParticle,rowCell,colCell;

  if (i<l)
  {
    for (size_t j = 0; j < N_PARTICLES; j++) {
      rowParticle = y_part_dev[j];
      colParticle = x_part_dev[j];
      rowCell = (i / WIDTH);
      colCell = (i % WIDTH);
      float distancia = (dist(rowParticle,colParticle,rowCell,colCell));
      if (distancia != -1) {
        map[i].charge += distancia;
      }
    }
  }
}

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

int main(int argc, char *argv[]){
  // Load data
  string input_file_name;

  if (argc > 1) {
		input_file_name = argv[1];
	} else {
		cout << "faltó un argumento" << endl;
		exit(0);
	}

	ifstream infile;
	cout << "Reading: " <<  input_file_name.c_str() << endl;
	infile.open(input_file_name.c_str());

  int nP;
	float *x_part, *y_part;

  infile >> nP;
  cout << "nP: "<<nP << endl;

  x_part = (float *)malloc(nP * sizeof(float));
	y_part = (float *)malloc(nP * sizeof(float));

  for (int i = 0; i<nP; i++) {
		infile >> x_part[i] >> y_part[i];
	}

  // Get memory for structures
  //float *cells, *d_cells,*outData,*out2,*out3,y[4];
  cell *cells, *d_cells, *dev_out, *dev_out2,*out;
  float *x_part_dev, *y_part_dev;
  //cells = (float*)malloc(WIDTH*LENGHT*sizeof(float));
  cells = (cell*)malloc(WIDTH*LENGHT*sizeof(cell));


  // Initialization grid with 0
  for (int i = 0; i < WIDTH*LENGHT; i++) {
    cells[i].charge = 0.0;
  }

  // Define sizes of GPU
  int blockSize = 256; // # threads
  int gridSize = ((WIDTH*LENGHT)/blockSize)+ ((WIDTH*LENGHT) % blockSize != 0); // # blocks
  int sharedBytes = blockSize*sizeof(cell);

  // Get memory in GPU for structures
  // data for charge function
  CUDA_CHECK(cudaMalloc(&d_cells, WIDTH*LENGHT*sizeof(cell))); // 1D array representation for grid 2D
  //CUDA_CHECK(cudaMalloc(&x_part_dev, N_PARTICLES*sizeof(float)));
  //CUDA_CHECK(cudaMalloc(&y_part_dev, N_PARTICLES*sizeof(float)));

  // data for reduction function
  CUDA_CHECK(cudaMalloc(&dev_out, gridSize*sizeof(cell)));
  CUDA_CHECK(cudaMalloc(&dev_out2, (gridSize/blockSize)*sizeof(cell)));
  out = (cell*)malloc((gridSize/blockSize)*sizeof(cell));

  // Copy data from CPU to GPU
  CUDA_CHECK(cudaMemcpy(d_cells, cells, WIDTH*LENGHT*sizeof(cell), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(x_part_dev, x_part, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(y_part_dev, y_part, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
  cudaMemcpyToSymbol(x_part_dev, x_part, N_PARTICLES * sizeof(float))
  cudaMemcpyToSymbol(y_part_dev, y_part, N_PARTICLES * sizeof(float))

  cudaEvent_t ct1, ct2;
  float dt, dt2;

  // time before kernel
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);

  // Charge grid
  charge<<<gridSize,blockSize>>>(WIDTH*LENGHT, d_cells);
  cudaDeviceSynchronize();

  //Time after charge kernel
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt, ct1, ct2);
  float time1 = dt;

  std::cout << "Time GPU computing cells charges: " << time1 << "[ms]" << std::endl;

  //CUDA_CHECK(cudaMemcpy(cells, d_cells, WIDTH*LENGHT*sizeof(float), cudaMemcpyDeviceToHost));
  //cudaDeviceSynchronize();


  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  // time before kernel min
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);

  // Search min load
  reduce0<<<gridSize,blockSize,sharedBytes>>>(d_cells, dev_out);
  cudaDeviceSynchronize();
  cout << "first reduction" << endl;
  reduce0<<<gridSize/blockSize,blockSize,sharedBytes>>>(dev_out, dev_out2);
  cudaDeviceSynchronize();
  cout << "second reduction" << endl;
  CUDA_CHECK(cudaMemcpy(out, dev_out2, gridSize/blockSize*sizeof(float), cudaMemcpyDeviceToHost));
  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  //Time after min kernel
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt2, ct1, ct2);
  float time2 = dt2;

  std::cout << "Time GPU computing minimum value: " << time2 << "[ms]" << std::endl;

  for (size_t i = 0; i < gridSize/blockSize; i++) {
    cout << out[i] << endl;
  }

  // Escribiendo resultado en archivo
  ofstream times_file;
  times_file.open("results_tarea_4_2.txt", ios_base::app);
  times_file << input_file_name.c_str() << endl;
  times_file << "Tiempo en charge kernel: "<< dt << "[ms]" << endl;
  times_file << "Tiempo en min kernel: "<< dt2 << "[ms]" << endl;

  cudaFree(d_cells);
  cudaFree(dev_out2);
  cudaFree(dev_out);
  free(cells);
  free(out)
  free(x_part);
  free(y_part);

  return 0;
}
