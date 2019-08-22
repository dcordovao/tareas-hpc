
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <ctime>
#include <iomanip>

using namespace std;

// Función auxiliar para guardar el nombre del archivo bien
vector<string> splitpath(const string& str, const set<char> delimiters)
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

void swap_array(int size, int x, float *in, float *out)
{
	for (int i = 0; i < size; i++)
	{
		out[i] = ((i/x)%2 == 0)? in[i +x]: in[i-x];
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
	float *r_in, *g_in, *b_in, *r_out, *g_out, *b_out;

	infile >> M >> N;

	cout << M << N << endl;

	size  = M*N;

	// Allocating arrays
	r_in = (float *)malloc(size * sizeof(float));
	g_in = (float *)malloc(size * sizeof(float));
	b_in = (float *)malloc(size * sizeof(float));

	r_out = (float *)malloc(size * sizeof(float));
	g_out = (float *)malloc(size * sizeof(float));
	b_out = (float *)malloc(size * sizeof(float));

	// Reading channels
	for (int i = 0; i < size; i++)
	{
		infile >> r_in[i];
	}
	for (int i = 0; i < size; i++)
	{
		infile >> g_in[i];
	}
	for (int i = 0; i < size; i++)
	{
		infile >> b_in[i];
	}


	// Preparando archivo donde iran los resultados
	set<char> delims{'/'};
	vector<string> path = splitpath(input_file_name, delims);
	ofstream times_file, result_file;
	times_file.open("resultados/times_cpp.txt", ios_base::app);
	

	int x_to_test[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

	for (int i = 0; i < 10; i++) 
	{
		int X = x_to_test[i];
		clock_t start;
		double duration;
		start = clock();

		// Llamar al algoritmo
		swap_array(size, X, r_in, r_out);
		swap_array(size, X, g_in, g_out);
		swap_array(size, X, b_in, b_out);

		duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		double duration_ms = duration*1000;
		
		cout<<"printf: "<< setprecision(4) << duration_ms << "[ms]" << '\n';

		
		// Escribiendo resultado en archivo
		times_file <<  "X = "<< X << " " << path.back() << " " << setprecision(4) << duration_ms << "[ms]" << endl;

		// Printing the result file
		
		string result_file_name = "resultados/result_cpp_x"+to_string(X)+"_"+path.back();
		cout << result_file_name << endl;
		result_file.open(result_file_name);

		result_file << M << " " << N << endl;
		for (int j = 0; j < size-1; j++)
		{
			result_file << r_out[j] << " ";
		}
		result_file << r_out[size-1] << endl;

		for (int j = 0; j < size-1; j++)
		{
			result_file << g_out[j] << " ";
		}
		result_file << g_out[size-1] << endl;

		for (int j = 0; j < size-1; j++)
		{
			result_file << b_out[j] << " ";
		}
		result_file << b_out[size-1];

		result_file.close();
	}
	

	

	

	// Liberar memoria
	free(r_in);
	free(g_in);
	free(b_in);
	free(r_out);
	free(g_out);
	free(b_out);
	times_file.close();
	infile.close();

	return 0;
}
