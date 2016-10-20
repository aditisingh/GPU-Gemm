#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>

using namespace std;



int main(int argc, char* argv[])
{
	if(argc != 4) //there should be four arguments
	return 1; //exit and return an error

	ifstream infile_A, infile_B;	//reading the input matrices
	
	//reading matrix A
	infile_A.open(argv[1]);
	string line;
	
	int count=0;
	int len=0;
	while(getline(infile_A,line)){
		len+=line.length();
		cout<<line.length()<<endl;;
		for (int i=0; i<=line.length();i++)
		{
			float val=line[i];
			cout<<val<<" ";
			count++;
		}
		cout<<endl<<endl;
	}
	
	cout<<"count="<<count<<" ";
	cout<<"len="<<len;

	return 0;
}
