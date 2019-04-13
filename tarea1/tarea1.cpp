
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <set>

using namespace std;

vector<string> splitpath(
  const string& str
  , const set<char> delimiters)
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

int main(int argc, char *argv[]){


    string input_file_name;

    if (argc > 1) {
	    input_file_name = argv[1];
	  }

    ifstream infile;
   	infile.open(input_file_name.c_str());

   	int L,M,N, rows, cols;
   	float **matrix, *r_out, *g_out, *b_out;

   	infile >> L >> M >> N;

   	rows = L*3;
   	cols = M*N;

   	// Allocating matrix
   	matrix = (float **)malloc(rows * sizeof(float *));
    for(int i = 0; i < rows; i++) matrix[i] = (float *)malloc(cols * sizeof(float));
  	r_out = (float *)malloc(cols * sizeof(float));
  	g_out = (float *)malloc(cols * sizeof(float));
  	b_out = (float *)malloc(cols * sizeof(float));
    // Initialize with zeros
    // I didn't use Calloc because it doesn't work with floats
    for (int j = 0; j < cols; j++)
    {
      r_out[j] = 0;
      g_out[j] = 0;
      b_out[j] = 0;
    }

	  // Reading matrix
   	for (int i = 0; i < rows; i++)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			infile >> matrix[i][j];
   		}
   	}

   	// Adding R, G and B channels
   	for (int i = 0; i < rows; i+=3)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			r_out[j] += matrix[i][j];
        g_out[j] += matrix[i+1][j];
        b_out[j] += matrix[i+2][j];
   		}
   	}
   	// Dividing by L de R, G and B Channels
   	for (int j = 0; j < cols; j++)
  	{
  		r_out[j] /= L;
      g_out[j] /= L;
      b_out[j] /= L;
  	}

    set<char> delims{'/'};
    vector<string> path = splitpath(input_file_name, delims);

    // Printing the result file
	  ofstream result_file;
   	result_file.open("result_"+path.back());

   	result_file << M << " " << N << endl;
   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << r_out[j] << " ";
   	}
   	result_file << r_out[cols-1] << endl;

   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << g_out[j] << " ";
   	}
   	result_file << g_out[cols-1] << endl;

   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << b_out[j] << " ";
   	}
   	result_file << b_out[cols-1];

   	// system("python3 converter.py 1 result");

    return 0;
}
