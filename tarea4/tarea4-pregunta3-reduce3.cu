#include <stdio.h>
#include <iostream>
#include <iostream>
#include <fstream>
#include <limits>

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



__global__ void add_charges(float l, cell *map)
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

__global__ void add_one_particle(int l, int part_pos, cell *map)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float rowParticle,colParticle,rowCell,colCell;

  if (i<l)
  {
    rowParticle = (part_pos / WIDTH); 
    colParticle = (part_pos % WIDTH);
    rowCell = (i / WIDTH);
    colCell = (i % WIDTH);
    float distancia = (dist(rowParticle,colParticle,rowCell,colCell));
    if (distancia != -1) {
      map[i].charge += distancia;
    }
  }
}

__global__ void reduce3(cell *g_idata, cell *g_odata)
{
    extern __shared__ cell sdata[];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = (g_idata[i].charge< g_idata[i+blockDim.x].charge)? g_idata[i]:g_idata[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) {
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
  cell *cells, *d_cells, *dev_out, *dev_out2, *dev_out3,*out;
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
  int gridSize = ((WIDTH*LENGHT)/blockSize)+ ((WIDTH*LENGHT) % blockSize != 0); // # blocks
  int sharedBytes = blockSize*sizeof(cell);

  cout << "Gridsize: " << gridSize << endl;

  // Get memory in GPU for structures
  // data for charge function
  CUDA_CHECK(cudaMalloc(&d_cells, WIDTH*LENGHT*sizeof(cell))); // 1D array representation for grid 2D
  //CUDA_CHECK(cudaMalloc(&x_part_dev, N_PARTICLES*sizeof(float)));
  //CUDA_CHECK(cudaMalloc(&y_part_dev, N_PARTICLES*sizeof(float)));

  // data for reduction function
  CUDA_CHECK(cudaMalloc(&dev_out, gridSize*sizeof(cell)));
  CUDA_CHECK(cudaMalloc(&dev_out2, (gridSize/blockSize)*sizeof(cell)));
  CUDA_CHECK(cudaMalloc(&dev_out3, (gridSize/blockSize)/blockSize*sizeof(cell)));
  out = (cell*)malloc((gridSize/blockSize)/blockSize*sizeof(cell));

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
  add_charges<<<gridSize,blockSize>>>(WIDTH*LENGHT, d_cells);
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


  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  // Copy AGAIN
  CUDA_CHECK(cudaMemcpy(d_cells, cells, WIDTH*LENGHT*sizeof(cell), cudaMemcpyHostToDevice));

  // time before kernel min
  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);

  int large_out_1, large_out_2, large_out_3;
  large_out_1 = gridSize;
  large_out_2 = large_out_1/blockSize;
  large_out_3 = large_out_2/blockSize;

  // Search min load
  reduce3<<<large_out_1/2,blockSize,sharedBytes>>>(d_cells, dev_out);
  cudaDeviceSynchronize();
  cout << "first reduction" << endl;
  reduce3<<<large_out_2/2, blockSize, sharedBytes>>>(dev_out, dev_out2);
  cudaDeviceSynchronize();
  cout << "second reduction" << endl;
  reduce3<<<large_out_3/2,blockSize,sharedBytes>>>(dev_out2, dev_out3);
  cudaDeviceSynchronize();
  cout << "third reduction" << endl;
  CUDA_CHECK(cudaMemcpy(out, dev_out3, ((gridSize/blockSize)/blockSize)*sizeof(cell), cudaMemcpyDeviceToHost));
  // check for errors
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
  }

  //Time after min kernel
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt2, ct1, ct2);
  float time2 = dt2;

  std::cout << "Time GPU computing minimum value: " << time2 << "[ms]" << std::endl;

  cell mejor_cell = out[0];
  for (size_t i = 0; i < large_out_3; i++) {
    mejor_cell = (out[i].charge < mejor_cell.charge)? out[i]: mejor_cell;
    cout << "Charge: " << out[i].charge << "   Pos: " << out[i].index << endl;
  }

  cell *best_1000_cells;
  best_1000_cells = (cell *)malloc(1000 * sizeof(cell));
  cell *inf_array;
  inf_array = (cell *)malloc(sizeof(cell));

  // Primera mejor celda
  best_1000_cells[0] = mejor_cell;
  float reduction_total_time = 0, add_charges_total_time = 0;

  for (int i = 1; i < 1000; i++)
  {

    float dt3;
    // time before kernel charge
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    add_one_particle<<<gridSize,blockSize>>>(WIDTH*LENGHT, best_1000_cells[i-1].index, d_cells);
    cudaDeviceSynchronize();
    
    //Time after charge kernel
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt3, ct1, ct2);

    add_charges_total_time += dt3;

    mejor_cell.charge = inf;
    inf_array[0] = mejor_cell;
    CUDA_CHECK(cudaMemcpy(&d_cells[mejor_cell.index], inf_array, sizeof(cell), cudaMemcpyHostToDevice));

    
    // time before min charge
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    // Search min load
    reduce3<<<large_out_1/2,blockSize,sharedBytes>>>(d_cells, dev_out);
    cudaDeviceSynchronize();
    reduce3<<<large_out_2/2, blockSize, sharedBytes>>>(dev_out, dev_out2);
    cudaDeviceSynchronize();
    reduce3<<<large_out_3/2,blockSize,sharedBytes>>>(dev_out2, dev_out3);
    cudaDeviceSynchronize();

    //Time after min kernel
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt3, ct1, ct2);

    reduction_total_time += dt3;

    CUDA_CHECK(cudaMemcpy(out, dev_out3, ((gridSize/blockSize)/blockSize)*sizeof(cell), cudaMemcpyDeviceToHost));
    

    mejor_cell = out[0];
    for (size_t j = 0; j < large_out_3; j++) {
      mejor_cell = (out[j].charge < mejor_cell.charge)? out[j]: mejor_cell;
    }

    best_1000_cells[i] = mejor_cell;
  }

  // Escribiendo resultado en archivo
  ofstream times_file;
  times_file.open("results_tarea_4_3_reduce3.txt");
  times_file << input_file_name.c_str() << endl;
  times_file << "Tiempo en charge kernel primera iteracion: "<< dt << "[ms]" << endl;
  times_file << "Tiempo en min kernel primera iteracion: "<< dt2 << "[ms]" << endl;
  times_file << "Tiempo en charge kernel 999 iteraciones: "<< add_charges_total_time << "[ms]" << endl;
  times_file << "Tiempo en min kernel 999 iteraciones: "<< reduction_total_time << "[ms]" << endl;

  times_file << "\n Resultados 1000 casillas (Carga, posicion): \n";

  for (int i = 0; i < 1000; i++)
  {
    times_file << best_1000_cells[i].charge << "    " << best_1000_cells[i].index << endl;
  }

  cudaFree(d_cells);
  cudaFree(dev_out);
  cudaFree(dev_out2);
  cudaFree(dev_out3);
  free(cells);
  free(out);
  free(x_part);
  free(y_part);

  return 0;
}
