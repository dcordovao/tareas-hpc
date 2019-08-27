#include <stdio.h>
#include <iostream>
#include <iostream>
#include <fstream>
#include <limits>

#define WIDTH 8192
#define LENGHT 8192
#define RADIO 100
#define SUBGRID 40401
#define PARTICLES_PER_THREAD 100
#define N_NEW 20200500
#define N_PARTICLES 5000

__constant__ float x_part_dev[N_PARTICLES];
__constant__ float y_part_dev[N_PARTICLES];

using namespace std;

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)

typedef struct
{
    float charge;
    int index;
} cell;

__device__ float dist(float x1, float y1, float x2, float y2)
{
  float dist;
  dist = sqrtf(powf(x2-x1, 2) + powf(y2-y1, 2));
  if(dist != 0) return 1/dist;
  else return -1;
}

__global__ void charge(cell *map)
{
  int x_cen,y_cen,subgrid_x,subgrid_y, index;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  //int id_particle_piv = idx/SUBGRID;
  int id_particle = idx/SUBGRID;
  float d;

  //if(id_particle_piv*PARTICLES_PER_THREAD < N_PARTICLES) 
  //{
    //for (size_t i = id_particle_piv*PARTICLES_PER_THREAD;
    //  i < id_particle_piv*PARTICLES_PER_THREAD + PARTICLES_PER_THREAD; i++)
    //{
  x_cen = x_part_dev[id_particle];
  y_cen = y_part_dev[id_particle];
  subgrid_x = (id_particle%SUBGRID)%201;
  subgrid_y = (id_particle%SUBGRID)/201;
  d = dist(x_cen,y_cen,subgrid_x,subgrid_y);
  index = ((y_cen-RADIO)+subgrid_y)*WIDTH+((x_cen-RADIO)+subgrid_x);
  if(0<=index && index<WIDTH*LENGHT) atomicAdd(&map[index].charge, d);
    //}
  //}
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
  float inf = std::numeric_limits<float>::infinity();

  infile >> nP;
  cout << "nP: "<<nP << endl;

  x_part = (float *)malloc(nP * sizeof(float));
	y_part = (float *)malloc(nP * sizeof(float));

  for (int i = 0; i<nP; i++) {
		infile >> x_part[i] >> y_part[i];
	}

  // Get memory for structures
  //float *cells, *d_cells,*outData,*out2,*out3,y[4];
  cell *cells, *d_cells;
  // HABIA QUE COMENTAR ESTO float *x_part_dev, *y_part_dev;
  //cells = (float*)malloc(WIDTH*LENGHT*sizeof(float));
  cells = (cell*)malloc(WIDTH*LENGHT*sizeof(cell));

  // Initialization grid with 0
  for (int i = 0; i < WIDTH*LENGHT; i++) {
    cells[i].charge = 0.0;
    cells[i].index = i;
  }

  // Define sizes of GPU
  int blockSize = 256; // # threads
  int gridSize = (SUBGRID*N_PARTICLES/blockSize)+(SUBGRID*N_PARTICLES % blockSize != 0); // # blocks

  cout << "Gridsize: " << gridSize << endl;

  // Get memory in GPU for structures
  // data for charge function
  CUDA_CHECK(cudaMalloc(&d_cells, WIDTH*LENGHT*sizeof(cell))); // 1D array representation for grid 2D

  // Copy data from CPU to GPU
  CUDA_CHECK(cudaMemcpy(d_cells, cells, WIDTH*LENGHT*sizeof(cell), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(x_part_dev, x_part, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaMemcpy(y_part_dev, y_part, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
  cudaMemcpyToSymbol(x_part_dev, x_part, N_PARTICLES * sizeof(float));
  cudaMemcpyToSymbol(y_part_dev, y_part, N_PARTICLES * sizeof(float));

  cudaEvent_t ct1, ct2;
  float dt, dt2;

  // time before kernel
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);

  // Charge grid
  charge<<<gridSize,blockSize>>>(d_cells);
  cudaDeviceSynchronize();

  //Time after charge kernel
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt, ct1, ct2);
  float time1 = dt;

  std::cout << "Time GPU computing cells charges: " << time1 << "[ms]" << std::endl;

  CUDA_CHECK(cudaMemcpy(cells, d_cells, WIDTH*LENGHT*sizeof(cell), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  int zeros=0;
  for(int i=0;i<WIDTH*LENGHT;i++)
  {
    if(cells[i].charge==0)
    {
      cells[i].charge = inf;
      zeros++;
    }
  }
  cout << "Zeros(Errors before charge: " << zeros << endl;

  cudaFree(d_cells);
  free(cells);
  free(x_part);
  free(y_part);
}
