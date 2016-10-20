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
	int length_A;
	//reading file A size
	
	infile_A.open(argv[1],ios::binary|ios::in|ios::ate);
	
	//get length of file
	infile_A.seekg(0,ios::end);
	length_A=infile_A.tellg();
	infile_A.seekg(0,ios::beg);
	
	//memory allocation
	matrix M;
	/*(cout<<length_A;
	vector<float> buffer((length_A*sizeof(char)-2*sizeof(unsigned int))/sizeof(float));

	size_t size=(length_A*sizeof(char)-2*sizeof(unsigned int));
	cout<<" "<<size<<" ";
	infile_A.read(&buffer[0],size/sizeof(float));
	
	cout<<buffer[0];*/
	//read data as block
	infile_A.read(reinterpret_cast<char*>(&M),8);
	cout<<M.rows<<M.cols;

	float* array=(float*)malloc(M.rows*M.cols*sizeof(float));
	//cout<<M.array;
	//infile_A.close();
	
	//cout.write((char*)buffer,length_A-1);
	
	//delete[] buffer;	
	
	return 0;
}
