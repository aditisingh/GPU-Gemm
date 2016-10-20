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

using namespace std;

struct matrix{
	unsigned int rows;
	unsigned int cols;
};

int main(int argc, char* argv[])
{
	if(argc != 4) //there should be four arguments
	return 1; //exit and return an error

	ifstream infile_A, infile_B;	//reading the input matrices
	//reading file A size
	
	infile_A.open(argv[1],ios::binary|ios::in|ios::ate);
	
	//get length of file
	infile_A.seekg(0,ios::end);
	infile_A.seekg(0,ios::beg);
	
	//memory allocation
	matrix M;
	infile_A.read(reinterpret_cast<char*>(&M),2*sizeof(unsigned int));
	cout<<M.rows<<M.cols;

	float* array_A=(float*)malloc(M.rows*M.cols*sizeof(float));
	infile_A.read(reinterpret_cast<char*>(array_A),M.rows*M.cols);
	
	infile_A.close();


	for(int i=0; i<M.rows*M.cols;i++)
		cout<<array_A[i]<<" ";
	
	infile_A.close();
		
	
	return 0;
}
