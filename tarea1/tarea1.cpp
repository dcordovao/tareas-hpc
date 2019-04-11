
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>

using namespace std;

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
	r_out = (float *)calloc(cols * sizeof(float)); // Initialize with zeros
	g_out = (float *)calloc(cols * sizeof(float)); // Initialize with zeros
	b_out = (float *)calloc(cols * sizeof(float)); // Initialize with zeros

	// Reading matrix
   	for (int i = 0; i < rows; i++)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			infile >> matrix[i][j];
   		}
   	}

   	// Adding R channels
   	for (int i = 0; i < rows; i+=3)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			r_out[j] += matrix[i][j];
   		}
   	}
   	// Dividing by L de R Channel
   	for (int j = 0; j < cols; j++)
	{
		r_out[j] /= L;
	}


	// Adding G channels
   	for (int i = 1; i < rows; i+=3)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			g_out[j] += matrix[i][j];
   		}
   	}
   	// Dividing by L de G Channel
   	for (int j = 0; j < cols; j++)
	{
		g_out[j] /= L;
	}

	// Adding B channels
   	for (int i = 2; i < rows; i+=3)
   	{
   		for (int j = 0; j < cols; j++)
   		{
   			b_out[j] += matrix[i][j];
   		}
   	}
   	// Dividing by L de B Channel
   	for (int j = 0; j < cols; j++)
	{
		b_out[j] /= L;
	}


	ofstream result_file;
   	result_file.open("result.txt");

   	result_file << M << " " << N << endl;
   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << r_out[j] << " ";
   	}
   	result_file << r_out[cols] << endl;

   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << g_out[j] << " ";
   	}
   	result_file << g_out[cols] << endl;

   	for (int j = 0; j < cols-1; j++)
   	{
   		result_file << b_out[j] << " ";
   	}
   	result_file << b_out[cols];

   	system("python converter.py 1 result.txt");


    //system("python converter.py 1 images/images1");

    return 0;
}