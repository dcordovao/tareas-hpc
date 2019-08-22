
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <ctime>
#include <iomanip>
#include <limits>
#include <cmath>

using namespace std;

#define GRID_WIDTH 8192
#define GRID_SIZE 67108864

int main(int argc, char *argv[]){


	string input_file_name;

	if (argc > 1) {
		input_file_name = argv[1];
	} else {
		cout << "faltó un argumento" << endl;
		exit(0);
	}

	ifstream infile;
	cout << input_file_name.c_str() << endl;
	infile.open(input_file_name.c_str());

	int N_PARTICULAS;
	float *grid, *x_part, *y_part;
	float inf = std::numeric_limits<float>::infinity();

	infile >> N_PARTICULAS;

	

	cout << "Cantidad de particulas en archivo:" <<  N_PARTICULAS << endl;

	grid = (float *)malloc(GRID_SIZE * sizeof(float));
	x_part = (float *)malloc(N_PARTICULAS * sizeof(float));
	y_part = (float *)malloc(N_PARTICULAS * sizeof(float));

	for (int i = 0; i<N_PARTICULAS; i++) {
		infile >> x_part[i] >> y_part[i];
	}

	int iteration = 1;
	
		
	float menor_suma = inf;
	int menor_index = -1;
	int menor_x = -1;
	int menor_y = -1;

	cout << "Itereacion: " << iteration << endl;
	for (int i=0; i < GRID_SIZE; i++) {
		int x = i%GRID_WIDTH;
		int y = i/GRID_WIDTH;
		float suma = 0;
		for (int j = 0; j < N_PARTICULAS; j++) {
			float dist = powf(x_part[j]-(float)x, 2) + powf(y_part[j]-(float)y, 2);       //calculating Euclidean distance
			dist = sqrtf(dist); 
			float q = 1/dist;
			suma += q;
			
		}
		grid[i] = suma;
		if (i<100) {
			cout << suma << endl;
		}
		if (suma < menor_suma) {
			menor_suma = suma;
			menor_index = i;
			menor_x = x;
			menor_y = y;
		}
	}

	cout << "Menor (X,Y): " << menor_x << " " << menor_y << endl;

	while(true) {
		iteration++;
		cout << "Itereacion: " << iteration << endl;
		break;
	}
}