
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <set>

using namespace std;

vector<string> splitpath( const string& str, const set<char> delimiters)
{
  vector<string> result;

  char const* pch = str.c_str();
  char const* start = pch;
  for(; *pch; ++pch)
  {
    if (delimiters.find(*pch) != delimiters.end())
    {
      if (start != pch)
      {
        string str(start, pch);
        result.push_back(str);
      }
      else
      {
        result.push_back("");
      }
      start = pch + 1;
    }
  }
  result.push_back(start);

  return result;
}

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      cout << cudaGetErrorString(error) << endl; \
    } \
  } while (0)

__global__ void kernel_colSum(float *r_in, float *g_in, float *b_in,
     float *r_result, float *g_result, float *b_result , int nrow, int ncol) {

    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (colIdx < ncol) {
        float sum_r=0;
        float sum_g=0;
        float sum_b=0;
        for (int k = 0 ; k < nrow ; k++) {
            sum_r+=r_in[colIdx+ncol*k];
            sum_g+=g_in[colIdx+ncol*k];
            sum_b+=b_in[colIdx+ncol*k];
        }
        r_result[colIdx] = sum_r;
        g_result[colIdx] = sum_g;
        b_result[colIdx] = sum_b;
    }
}

__global__ void kernel_colDiv(float *m, float *s, int nrow, int ncol) {

    int rowIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (rowIdx < nrow) {
        float sum=0;
        for (int k = 0 ; k < ncol ; k++)
            sum+=m[rowIdx*ncol+k];
        s[rowIdx] = sum;            
    }
}

int main(int argc, char *argv[]){

    string input_file_name;

    if (argc > 1) {
	    input_file_name = argv[1];
	}

    ifstream infile;
   	infile.open(input_file_name.c_str());

   	int L,M,N, rows, cols, total_pixels;
   	float *r_host, *g_host, *b_host, *r_out_host, *g_out_host, *b_out_host;
    float *r_dev, *g_dev, *b_dev, *r_out_dev, *g_out_dev, *b_out_dev;
   	infile >> L >> M >> N;

   	rows = L;
    cols = M*N;
    total_pixels = rows*cols;

   	// Allocating matrix
   	r_host = (float *)malloc(total_pixels * sizeof(float));
  	g_host = (float *)malloc(total_pixels * sizeof(float));
  	b_host = (float *)malloc(total_pixels * sizeof(float));
  	r_out_host = (float *)malloc(cols * sizeof(float));
  	g_out_host = (float *)malloc(cols * sizeof(float));
  	b_out_host = (float *)malloc(cols * sizeof(float));
    // Initialize with zeros
    // I didn't use Calloc because it doesn't work with floats
    for (int j = 0; j < cols; j++)
    {
      r_out_host[j] = 0.5;
      g_out_host[j] = 0.5;
      b_out_host[j] = 0.5;
    }

	  // Reading matrix
   	for (int i = 0; i < rows; i++)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			infile >> r_host[i*cols+j];
        }
           for (int j = 0; j < cols; j++)
   		{
   			infile >> g_host[i*cols+j];
        }
           for (int j = 0; j < cols; j++)
   		{
   			infile >> b_host[i*cols+j];
   		}
   	}
       
    cudaEvent_t ct1, ct2;
	double ms;
    float dt;
    
    CUDA_CHECK(cudaMalloc((void**)&r_dev, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_dev, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_dev, total_pixels * sizeof(float)));

    // Input matrix of images
    CUDA_CHECK(cudaMemcpy(r_dev, r_host, total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_dev, g_host, total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev, b_host, total_pixels * sizeof(float), cudaMemcpyHostToDevice));

    // Output image
    CUDA_CHECK(cudaMalloc((void**)&r_out_dev, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_out_dev, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_out_dev, cols * sizeof(float)));

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    int grid_size, block_size = 256;
    grid_size = (int)ceil((float) L * M * N / block_size);
    kernel_colSum<<<grid_size, block_size>>>(r_dev, g_dev, b_dev, r_out_dev, g_out_dev, b_out_dev, rows, cols);
       
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;

    CUDA_CHECK(cudaMemcpy(r_out_host, r_out_dev, cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_out_host, g_out_dev, cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_out_host, b_out_dev, cols * sizeof(float), cudaMemcpyDeviceToHost));

   	// Dividing by L de R, G and B Channels
   	for (int j = 0; j < cols; j++)
  	{
  		r_out_host[j] /= L;
        g_out_host[j] /= L;
        b_out_host[j] /= L;
  	}

    set<char> delims{'/'};
    vector<string> path = splitpath(input_file_name, delims);

    // Printing the result file
	  ofstream result_file;
   	result_file.open("result_cuda_"+path.back());

   	result_file << M << " " << N << endl;
   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << r_out_host[j] << " ";
   	}
   	result_file << r_out_host[cols-1] << endl;

   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << g_out_host[j] << " ";
   	}
   	result_file << g_out_host[cols-1] << endl;

   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << b_out_host[j] << " ";
   	}
   	result_file << b_out_host[cols-1];

   	// system("python3 converter.py 1 result");

    return 0;
}
