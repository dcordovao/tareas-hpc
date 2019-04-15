#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)

/*  Read file */

void Read(float** R, float** G, float** B,
	      int *M, int *N, int *L, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d %d\n", L, M, N);

		int rows,cols;
		rows = (*L)*3;
		cols = (*M) * (*N);
    int imsize = rows*cols;
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
		float* B1 = new float[imsize];

		numImg = 0;
		while(ind <imsize){
			for(int i = 0; i < cols; i++)
				fscanf(fp, "%f ", &(R1[i+(cols*numImg)]));
			for(int i = 0; i < cols; i++)
				fscanf(fp, "%f ", &(G1[i+(cols*numImg)]));
			for(int i = 0; i < cols; i++)
				fscanf(fp, "%f ", &(B1[i+(cols*numImg)]));
			numImg += 1;
		}
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
}

/* write file */
void Write(float* R, float* G, float* B,
	       int M, int N, const char *filename, int L) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[M*N-1]);
    fclose(fp);
}

/* Procesamiento Imagen GPU */
// Rdev,Gdev,Bdev,L,M,N,Rdevout,Gdevout,Bdevout
__global__ void kernelGPU(float *inR, float *inG, float *inB, cont int L, const int M,
													const int N, float *outR, float *outG, float *outB)
{
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
  
}

/* main */
int main(int argc, char **argv){
	cudaEvent_t ct1, ct2;
	double ms;
	float dt;
	int L,M, N;
  float *Rhost, *Ghost, *Bhost;
  float *Rhostout, *Ghostout, *Bhostout;
  float *Rdev, *Gdev, *Bdev;
  float *Rdevout, *Gdevout, *Bdevout;

  Read(&Rhost, &Ghost, &Bhost, &M, &N, "img.txt", &L);

  /* GPU */

  int grid_size, block_size = 256;
  grid_size = (int)ceil((float) L * M * N / block_size);

  CUDA_CHECK(cudaMalloc((void**)&Rdev, L * M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&Gdev, L * M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&Bdev, L * M * N * sizeof(float)));

	// Input matrix of images
  CUDA_CHECK(cudaMemcpy(Rdev, Rhost, L * M * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Gdev, Ghost, L * M * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Bdev, Bhost, L * M * N * sizeof(float), cudaMemcpyHostToDevice));

	// Output image
  CUDA_CHECK(cudaMalloc((void**)&Rdevout, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&Gdevout, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&Bdevout, M * N * sizeof(float)));

  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);
  kernel<<<grid_size, block_size>>>(Rdev,Gdev,Bdev,L,M,N,Rdevout,Gdevout,Bdevout);
  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&dt, ct1, ct2);
  std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;

  Rhostout = new float[M*N];
  Ghostout = new float[M*N];
  Bhostout = new float[M*N];
  CUDA_CHECK(cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  Write(Rhostout, Ghostout, Bhostout, M, N, "imgGPU.txt");

	// Free memory
  CUDA_CHECK(cudaFree(Rdev));
	CUDA_CHECK(cudaFree(Gdev));
	CUDA_CHECK(cudaFree(Bdev));
  CUDA_CHECK(cudaFree(Rdevout));
	CUDA_CHECK(cudaFree(Gdevout));
	CUDA_CHECK(cudaFree(Bdevout));

  delete[] Rhost;
	delete[] Ghost;
	delete[] Bhost;
  delete[] Rhostout;
	delete[] Ghostout;
	delete[] Bhostout;

return 0;
}
