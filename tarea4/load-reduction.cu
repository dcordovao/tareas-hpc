#include <stdio.h>
#include <iostream>
#include <iostream>
#include <fstream>

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


__device__ float dist(float x1, float y1, float x2, float y2)
{
  float dist;
  dist = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
  //dist = sqrtf(powf(x2-x1, 2) + powf(y2-y1, 2));
  if(dist != 0) return 1/dist;
  else return -1;
}

__global__ void charge(float l, float *map,float *X,float *Y)
{
  
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float rowParticle,colParticle,rowCell,colCell;

  for (int i = idx*CELLS_FOR_THREAD; i<idx*CELLS_FOR_THREAD+CELLS_FOR_THREAD; i++)
  {
    if (i<l)
    {
      for (size_t j = 0; j < N_PARTICLES; j++) {
        rowParticle = Y[j];
        colParticle = X[j];
        rowCell = (i / WIDTH);
        colCell = (i % WIDTH);
        //float distancia = rowCell-colCell;
        float distancia = 1;//(dist(rowParticle,colParticle,rowCell,colCell);
        if (distancia != -1) {
          map[i] += distancia;
        }
      }
      //map[i] = 1;
    }
  }
}

__global__
void chargeWithRadio(int l, float *map,float *X,float *Y)
{
  float d;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int rowPartcile,colParticle,rowCell,colCell;

  if (idx < l)
  {
    for (size_t i = 0; i < N_PARTICLES; i++) {
      rowPartcile = Y[i];
      colParticle = X[i];
      rowCell = (idx / WIDTH)+1;
      colCell = (idx % WIDTH)+1;
      d = dist(rowPartcile,colParticle,rowCell,colCell);
      map[idx] += (d<RADIO)?d:0.0;
    }
  }
}

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
  float *cells, *d_cells,*outData,*out2,*out3,y[4];
  float *x_part_dev, *y_part_dev;
  cells = (float*)malloc(WIDTH*LENGHT*sizeof(float));
  

  // Initialization grid with 0
  for (int i = 0; i < WIDTH*LENGHT; i++) {
    cells[i] = 0.0;
  }

  // Define sizes of GPU
  int blockSize = 256; // # threads
  int gridSize = ((WIDTH*LENGHT)/256)/CELLS_FOR_THREAD; // # blocks

  cout << "gridSize: " << gridSize << endl; 
  // Get memory in GPU for structures

  // data for charge function
  //cudaMalloc(&x_dev, nP * sizeof(float)); // X cord for particles
  //cudaMalloc(&y_dev, nP * sizeof(float)); // Y cord for particles
  CUDA_CHECK(cudaMalloc(&d_cells, WIDTH*LENGHT*sizeof(float))); // 1D array representation for grid 2D
  CUDA_CHECK(cudaMalloc(&x_part_dev, N_PARTICLES*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_part_dev, N_PARTICLES*sizeof(float)));

  // data for reduction function
  CUDA_CHECK(cudaMalloc(&outData, gridSize*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out2, (gridSize/blockSize)*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out3, ((gridSize/blockSize)/blockSize)*sizeof(float)));

  // Copy data from CPU to GPU
  CUDA_CHECK(cudaMemcpy(d_cells, cells, WIDTH*LENGHT*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(x_part_dev, x_part, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y_part_dev, y_part, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
  //cudaMemcpy(x_dev, &x_part,  nP * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(y_dev, &y_part,  nP * sizeof(float), cudaMemcpyHostToDevice);


  cudaEvent_t ct1, ct2;
  float dt, dt2;

  // time before kernel
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);

  // Charge grid
  charge<<<gridSize,blockSize>>>(WIDTH*LENGHT, d_cells, x_part_dev, y_part_dev); 
  cudaDeviceSynchronize();

  //Time after charge kernel
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt, ct1, ct2);
  float time1 = dt;

  std::cout << "Time GPU computing cells charges: " << time1 << "[ms]" << std::endl;

  CUDA_CHECK(cudaMemcpy(cells, d_cells, WIDTH*LENGHT*sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  
  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  for (size_t i = 0; i < 100; i++) {
    cout << cells[i] << ' ';
  }
  
  cout << endl;
  float suma = 0;
  for (int i = 0; i < WIDTH*LENGHT; i++) {
    if (cells[i] == 0)
    {
      cout << "i: " << i << " = 0"<< endl;
      break;
    }
    suma += cells[i];
  }
  cout << "Suma: " << suma << endl;

  cout << "\n \n primera parte exitosa (?)" << endl;

  // time before kernel min
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);

  // Search min load
  minReduction<<<gridSize,blockSize>>>(d_cells,outData); // First reduction 8192*8192 -> (8192*8192+255)/ 256 = 262.144
  cudaDeviceSynchronize();
  minReduction<<<gridSize/blockSize,blockSize>>>(outData,out2); // Second reduction 262.144 -> 262.144/256 = 1024
  cudaDeviceSynchronize();
  minReduction<<<(gridSize/blockSize)/blockSize,blockSize>>>(out2,out3); // Third reduction 262.144 -> 4 :)
  cudaDeviceSynchronize();

  //Time after min kernel
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt2, ct1, ct2);
  float time2 = dt2;

  std::cout << "Time GPU computing minimum value: " << time2 << "[ms]" << std::endl;

  // check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  // Escribiendo resultado en archivo
  ofstream times_file;
  times_file.open("results_tarea_4_2.txt", ios_base::app);
  times_file << input_file_name.c_str() << endl;
  times_file << "Tiempo en charge kernel: "<< dt << "[ms]" << endl;
  times_file << "Tiempo en min kernel: "<< dt2 << "[ms]" << endl;

  cudaMemcpy(y, out3, 4*sizeof(float), cudaMemcpyDeviceToHost);

  int min=INF;
  // min load
  for (size_t i = 0; i < 4; i++) {
    min = (y[i]<min)?y[i]:min;
  }

  cout << min << endl;

  //cudaFree(x_dev);
  //cudaFree(y_dev);
  cudaFree(d_cells);
  cudaFree(outData);
  cudaFree(out2);
  cudaFree(out3);
  free(cells);
  free(x_part);
  free(y_part);

  return 0;
}
