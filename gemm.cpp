#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <ctime>

using namespace std;

struct matrix{
	unsigned int rows;
	unsigned int cols;
};

float* matrix_mult(float* array1, unsigned int rows1, unsigned int cols1, float* array2, unsigned int rows2, unsigned int cols2)
{
	float* C=(float*)malloc(rows1*cols2*sizeof(float));
	
	//initailize the array to zero
	for(int idx=0; idx<rows1*cols2;idx++)
	{
		C[idx]=0;
		int c=(int)(idx/rows1);
		int r=idx%rows1;

		for(int k=0;k<rows2;k++)
		{
			C[idx]+=array1[rows1*k+r]*array2[rows2*c+k];
		}

	}	
	
	return C;

}

int main(int argc, char* argv[])
{
	if(argc != 4) //there should be four arguments
	return 1; //exit and return an error

	time_t reading_start=time(NULL);

	ifstream infile_A, infile_B;	//reading the input matrices
	
	
	//READING matrix A
	infile_A.open(argv[1],ios::binary|ios::in|ios::ate);
	
	//getting end and beginning of the file
	infile_A.seekg(0,ios::end);
	infile_A.seekg(0,ios::beg);
	
	//memory allocation
	matrix M_A;
	infile_A.read(reinterpret_cast<char*>(&M_A),2*sizeof(unsigned int));
	//cout<<M_A.rows<<M_A.cols;
	
	float* array_A=(float*)malloc(M_A.rows*M_A.cols*sizeof(float));	//column major
	infile_A.read(reinterpret_cast<char*>(array_A),M_A.rows*M_A.cols);
	
	infile_A.close();

	//READING matrix B
	infile_B.open(argv[1],ios::binary|ios::in|ios::ate);
	
	//getting end and beginning of the file
	infile_B.seekg(0,ios::end);
	infile_B.seekg(0,ios::beg);
	
	//memory allocation
	matrix M_B;
	infile_B.read(reinterpret_cast<char*>(&M_B),2*sizeof(unsigned int));
	//cout<<M_B.rows<<M_B.cols;

	float* array_B=(float*)malloc(M_B.rows*M_B.cols*sizeof(float));	//column major
	infile_B.read(reinterpret_cast<char*>(array_B),M_B.rows*M_B.cols);
	
	infile_B.close();

	time_t reading_end = time(NULL);

	float* array_C=matrix_mult(array_A,M_A.rows,M_A.cols,array_B,M_B.rows,M_B.cols);

	// for(int i=0; i<M_A.rows*M_B.cols;i++)
		// cout<<array_C[i]<<" ";

	time_t mult_end = time(NULL);

	//SAVING THE OUTPUT MATRIX
	ofstream ofile(argv[3], ios::binary);

	ofile.write((char*) &M_A.rows, sizeof(unsigned int));
	ofile.write((char*) &M_B.cols, sizeof(unsigned int));	
	ofile.write((char*) array_C , M_A.rows*M_B.cols*sizeof(float))	;

	time_t saved = time(NULL);

	cout<<"Matrix reading     :"<<double(reading_end - reading_start)<<" secs"<<endl;
	cout<<"Multiplication done:"<<double(mult_end - reading_end)<<" secs"<<endl;
	cout<<"Matrix saving      :"<<double(saved - mult_end)<<" secs"<<endl;

	return 0;
}
