
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

// Sumar cada columna(pixel) de las imagenes en paralelo
__global__ void kernel_swapArray(float *r_in, float *g_in, float *b_in,
  float *r_result, float *g_result, float *b_result , int size, int x) {

  int Idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (Idx < size) {
    r_result[Idx] = ((Idx/x)%2 == 0)? r_in[Idx +x]: r_in[Idx-x];
    g_result[Idx] = ((Idx/x)%2 == 0)? g_in[Idx +x]: g_in[Idx-x];
    b_result[Idx] = ((Idx/x)%2 == 0)? b_in[Idx +x]: b_in[Idx-x];
  }
}

int main(int argc, char *argv[]){


	string input_file_name;

	if (argc > 1) {
		input_file_name = argv[1];
	}

	ifstream infile;
	cout << input_file_name.c_str() << endl;
	infile.open(input_file_name.c_str());

	int M,N, size;
  float *r_in_host, *g_in_host, *b_in_host, *r_out_host, *g_out_host, *b_out_host;
  float *r_in_dev, *g_in_dev, *b_in_dev, *r_out_dev, *g_out_dev, *b_out_dev;

	infile >> M >> N;

	cout << M << N << endl;

	size  = M*N;

	// Allocating arrays
	r_in_host = (float *)malloc(size * sizeof(float));
	g_in_host = (float *)malloc(size * sizeof(float));
	b_in_host = (float *)malloc(size * sizeof(float));

	r_out_host = (float *)malloc(size * sizeof(float));
	g_out_host = (float *)malloc(size * sizeof(float));
	b_out_host = (float *)malloc(size * sizeof(float));

	// Reading channels
	for (int i = 0; i < size; i++)
	{
		infile >> r_in_host[i];
	}
	for (int i = 0; i < size; i++)
	{
		infile >> g_in_host[i];
	}
	for (int i = 0; i < size; i++)
	{
		infile >> b_in_host[i];
	}


	// Preparando archivo donde iran los resultados
	set<char> delims{'/'};
	vector<string> path = splitpath(input_file_name, delims);
	ofstream times_file, result_file;
	times_file.open("resultados/times_cuda_pregunta2.txt", ios_base::app);
	

	int x_to_test[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

	for (int i = 0; i < 10; i++) 
	{
    int X = x_to_test[i];
    
    cudaEvent_t ct1, ct2;
    float dt;
    

    // Input in device
    CUDA_CHECK(cudaMalloc((void**)&r_in_dev, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_in_dev, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_in_dev, size * sizeof(float)));

		// Copy
    CUDA_CHECK(cudaMemcpy(r_in_dev, r_in_host, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_in_dev, g_in_host, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_in_dev, b_in_host, size * sizeof(float), cudaMemcpyHostToDevice));
  
    // Output in device
    CUDA_CHECK(cudaMalloc((void**)&r_out_dev, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_out_dev, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&b_out_dev, size * sizeof(float)));

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    // Llamar algoritmo
    int grid_size, block_size = 256;
    grid_size = (int)ceil((float) size / block_size);
    kernel_swapArray<<<grid_size, block_size>>>(r_in_dev, g_in_dev, b_in_dev, r_out_dev, g_out_dev, b_out_dev, size, X);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);

    float duration;

		duration = dt;
    std::cout << "Tiempo GPU: " << duration << "[ms]" << std::endl;

    CUDA_CHECK(cudaMemcpy(r_out_host, r_out_dev, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_out_host, g_out_dev, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_out_host, b_out_dev, size * sizeof(float), cudaMemcpyDeviceToHost));

		// Escribiendo resultado en archivo
		times_file <<  "X = "<< X << " " << path.back() << " " << duration << "[ms]" << endl;

		// Printing the result file
		
		string result_file_name = "resultados/result_cuda_pregunta3_x"+to_string(X)+"_"+path.back();
		cout << result_file_name << endl;
		result_file.open(result_file_name);

		result_file << M << " " << N << endl;
		for (int j = 0; j < size-1; j++)
		{
			result_file << r_out_host[j] << " ";
		}
		result_file << r_out_host[size-1] << endl;

		for (int j = 0; j < size-1; j++)
		{
			result_file << g_out_host[j] << " ";
		}
		result_file << g_out_host[size-1] << endl;

		for (int j = 0; j < size-1; j++)
		{
			result_file << b_out_host[j] << " ";
		}
		result_file << b_out_host[size-1];

    result_file.close();

    CUDA_CHECK(cudaFree(r_in_dev));
    CUDA_CHECK(cudaFree(g_in_dev));
    CUDA_CHECK(cudaFree(b_in_dev));
    CUDA_CHECK(cudaFree(r_out_dev));
    CUDA_CHECK(cudaFree(g_out_dev));
    CUDA_CHECK(cudaFree(b_out_dev));
  }
  
	// Liberar memoria
	free(r_in_host);
	free(g_in_host);
	free(b_in_host);
	free(r_out_host);
	free(g_out_host);
	free(b_out_host);
	times_file.close();
	infile.close();

	return 0;
}
