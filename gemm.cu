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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

using namespace std;

struct matrix{
	unsigned int rows;
	unsigned int cols;
};

static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout<<cudaGetErrorString(err)<<" in "<< file <<" at line "<< line<<endl;
	}
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void matrix_mult(float* array1, unsigned int rows1, unsigned int cols1, float* array2, unsigned int rows2, unsigned int cols2, float* array3)
{
	 extern __shared__ float S[];			//defined a shared memory pointer
	
	size_t c=blockIdx.x*blockDim.x + threadIdx.x;
	size_t r=blockIdx.y*blockDim.y + threadIdx.y;

	if(r>=rows1 || c>=cols2) return;

	size_t idx=c*cols2+r;	//going columnwise
	size_t C1=rows2*c;
	
	float val=0;

	for(int i=0; i<rows1*cols1;i++)
		S[i]=array1[i];

	for(int i=0; i<rows2*cols2;i++)
		S[i+rows1*cols1]=array2[i];

	#pragma unroll 8
	for(int k=0;k<rows2;k++)
	{
		val+=S[rows1*k+r]*S[rows1*cols1+C1+k];
	}
	array3[idx]=val;

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
	
	//matrix M_A,M_B;

	float* array_A=(float*)malloc(M_A.rows*M_A.cols*sizeof(float));	//column major
	infile_A.read(reinterpret_cast<char*>(array_A),M_A.rows*M_A.cols*sizeof(float));
	
	infile_A.close();

	//READING matrix B
	infile_B.open(argv[1],ios::binary|ios::in|ios::ate);
	
	//getting end and beginning of the file
	infile_B.seekg(0,ios::end);
	infile_B.seekg(0,ios::beg);
	
	//memory allocation
	matrix M_B;
	infile_B.read(reinterpret_cast<char*>(&M_B),2*sizeof(unsigned int));

	float* array_B=(float*)malloc(M_B.rows*M_B.cols*sizeof(float));	//column major

/*	array_A[0]=1, array_A[3]=2, array_A[6]=1;
	array_A[1]=2, array_A[4]=3, array_A[7]=4;
	array_A[2]=1, array_A[5]=-1, array_A[8]=0;

	array_B[0]=0, array_B[3]=1, array_B[6]=0;
	array_B[1]=1, array_B[4]=2, array_B[7]=3;
	array_B[2]=-1, array_B[5]=2, array_B[8]=-1;

	*/

	infile_B.read(reinterpret_cast<char*>(array_B),M_B.rows*M_B.cols*sizeof(float));
	
	infile_B.close();

	if(M_A.cols!=M_B.rows)
	{
		cout<<"Illegal matrix sizes: "<<M_A.cols<<" != "<<M_B.rows<<endl;
		return 1;
	}

	float* array_C=(float*)malloc(M_A.rows*M_B.cols*sizeof(float));//gpu result
	
	float* array_D=(float*)malloc(M_A.rows*M_B.cols*sizeof(float));//cublas result
	
	time_t reading_end = time(NULL);


	//GPU DEVICE PROPERTIES
	int nDevices;
	HANDLE_ERROR(cudaGetDeviceCount(&nDevices));

	cudaDeviceProp prop;
   	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));	//using GPU0

   	//BLOCK AND GRID SIZE
    float thread_block=sqrt(prop.maxThreadsPerBlock);
	dim3 DimGrid(ceil(M_B.cols/thread_block),ceil(M_A.rows/thread_block),1); //image saved as a 2D grid
	dim3 DimBlock(thread_block,thread_block,1);

	size_t Sbytes = DimBlock.x * DimBlock.y * sizeof(float) * 2;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	if(props.sharedMemPerBlock < Sbytes){
		std::cout<<"ERROR: insufficient shared memory"<<std::endl;
		exit(1);
	}

	//GPU MEMORY ALLOCATION
	float *array_A_gpu, *array_B_gpu, *array_C_gpu, *array_D_gpu;
   
	HANDLE_ERROR(cudaMalloc(&array_A_gpu,M_A.rows*M_A.cols*sizeof(float))); //allocate space to store convolution result

	HANDLE_ERROR(cudaMalloc(&array_B_gpu,M_B.rows*M_B.cols*sizeof(float))); //allocate space to store convolution temporary

	HANDLE_ERROR(cudaMalloc(&array_C_gpu,M_A.rows*M_B.cols*sizeof(float))); //allocate space to copy image to GPU memory

	HANDLE_ERROR(cudaMalloc(&array_D_gpu,M_A.rows*M_B.cols*sizeof(float))); //allocate space to copy image to GPU memory


	//COPY TO GPU MEMORY
	HANDLE_ERROR(cudaMemcpy(array_A_gpu, array_A, M_A.rows*M_A.cols*sizeof(float), cudaMemcpyHostToDevice));//copy input image from global to gpu

	HANDLE_ERROR(cudaMemcpy(array_B_gpu, array_B, M_B.rows*M_B.cols*sizeof(float), cudaMemcpyHostToDevice));//copy the kernel0 host to device

	HANDLE_ERROR(cudaMemcpy(array_C_gpu, array_C, M_A.rows*M_B.cols*sizeof(float), cudaMemcpyHostToDevice));//copy kernel1 host to device

	HANDLE_ERROR(cudaMemcpy(array_D_gpu, array_D, M_A.rows*M_B.cols*sizeof(float), cudaMemcpyHostToDevice));//copy kernel1 host to device

	time_t memory_transfers=time(NULL);

	//time measurement
	cudaEvent_t start1, stop1;
 	
 	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	
	//MATRIX MULTIPLICATION
	cudaEventRecord(start1);
	matrix_mult<<<DimGrid, DimBlock, Sbytes>>>(array_A_gpu,M_A.rows,M_A.cols,array_B_gpu,M_B.rows,M_B.cols,array_C_gpu);
	cudaEventRecord(stop1);

	time_t mult_end = time(NULL);

	cudaEventSynchronize(stop1);
	float milliseconds1 = 0, milliseconds2 = 0;
	
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	cout<<"time taken by GPU = "<<milliseconds1<<" ms"<<endl;

	//copy to CPU MEMORY
	HANDLE_ERROR(cudaMemcpy(array_C, array_C_gpu, M_A.rows*M_B.cols*sizeof(float), cudaMemcpyDeviceToHost));//copy kernel1 host to device

	//Creating handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);	

	float alpha = 1.0;
	float beta = 0.0;
    
    cudaEvent_t start2, stop2;
 	
 	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	cudaEventRecord(start2);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M_A.rows, M_B.cols, M_A.cols, &alpha, array_A_gpu, M_A.rows, array_B_gpu, M_B.rows, &beta, array_D_gpu, M_A.rows);
	cudaEventRecord(stop2);

	cudaEventSynchronize(stop2);

	cudaEventElapsedTime(&milliseconds2, start2, stop2);
	cout<<"time taken by CUBLAS= "<<milliseconds2<<" ms"<<endl;
	
	//copy to CPU MEMORY
    
    HANDLE_ERROR(cudaMemcpy(array_D, array_D_gpu, M_A.rows*M_B.cols*sizeof(float), cudaMemcpyDeviceToHost));//copy kernel1 host to device

	float mse=0; //mean squared error

	for(int i=0; i<M_A.rows*M_B.cols;i++)
		{
		mse=mse+(array_C[i]-array_D[i])*(array_C[i]-array_D[i]);
		//float diff=array_C[i]-array_D[i];
		//cout<<diff<<" ";//
		// cout<<array_C[i]<<" "<<" "<<array_D[i]<<endl;
		}

	cout<<endl<<"Mean square error = "<<mse<<endl;

	//SAVING THE OUTPUT MATRIX
	ofstream ofile(argv[3], ios::binary);

	ofile.write((char*) &M_A.rows, sizeof(unsigned int));
	ofile.write((char*) &M_B.cols, sizeof(unsigned int));	
	ofile.write((char*) array_C , M_A.rows*M_B.cols*sizeof(float))	;

	time_t saved = time(NULL);

	//cout<<"Matrix reading     :"<<double(reading_end - reading_start)<<" secs"<<endl;
	//cout<<"Memory Transfers   :"<<double(memory_transfers - reading_end)<<" secs"<<endl;
	//cout<<"Multiplication done:"<<double(mult_end - memory_transfers)<<" secs"<<endl;
	//cout<<"Matrix saving      :"<<double(saved - mult_end)<<" secs"<<endl;

	return 0;
}
