/*
Parallel Prefix Sum(Scan) with CUDA
Hillis & Steele Algorithm

Yuechao Lu
2015.03.16
*/

// System includes 
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// CUDA runtime
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

void randomInit(float *data, int size);
__global__ void prescan(float *g_odata, float *g_idata, int dataSize);

int main()
{
	float *hostA,*hostB,*deviceA,*deviceB;
	unsigned int test_size = 8;


	hostA = (float *)malloc(sizeof(float)*test_size);
	hostB = (float *)malloc(sizeof(float)*test_size);

	cudaMalloc((void **)&deviceA, test_size * sizeof(float));
	cudaMalloc((void **)&deviceB, test_size * sizeof(float));

	// initialize host memory
	randomInit(hostA, test_size);
	cout << "Input data:" << endl;
	for (int i = 0; i < test_size; i++){
		cout << hostA[i] << "	";
	}
	cout << endl;
	/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from host memory A to device memory A ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
	cudaMemcpy(deviceA, hostA, test_size * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(1);
	dim3 dimBlock(test_size);
	//Be careful 2 buffer!!!
	prescan <<< dimGrid, dimBlock, 2 * test_size  * sizeof(float) >>>(deviceB, deviceA, test_size);

	/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from host memory A to device memory A ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
	cudaMemcpy(hostB, deviceB, test_size * sizeof(float), cudaMemcpyDeviceToHost);

	cout << "Output data:" << endl;
	for (int i = 0; i < test_size; i++){
		cout << hostB[i] << "	";
	}
	cout << endl;

	free(hostA);
	free(hostB);
	cudaFree(deviceA);
	cudaFree(deviceB);


	/*stop the console to see the results */
	cout << endl <<"The end."; char str1[1];	scanf("%s", &str1);
	return 0;
}

void randomInit(float *data, int size){
	for (int i = 0; i < size; ++i){
		data[i] = (i + 1);
	}
}

/*because it's parallel computing, you have to use double to avoid conflict*/
__global__ void prescan(float *g_odata, float *g_idata, int n)
{
	extern __shared__ float temp[]; // allocated on invocation  

	int thid = threadIdx.x;
	int pout = 0, pin = 1;

	// and set first element to 0  
	temp[thid] = g_idata[thid];	// Load input into first buffer.  
	temp[n + thid] = 0.0f;	   	// Second buffer all set to 0;
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2){
		pout = 1 - pout; // swap double buffer indices  
		pin = 1 - pout;
		if (thid >= offset) {
			temp[pout*n + thid] = temp[pin*n + thid] + temp[pin*n + thid - offset];
		} else {
			temp[pout*n + thid] = temp[pin*n + thid];
		}
	   __syncthreads();
	}
	g_odata[thid] = temp[pout*n + thid];
}
