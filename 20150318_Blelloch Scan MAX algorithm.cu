/*
Parallel Prefix MAX(EXCLUSIVE Scan) with CUDA
Blelloch Algorithm

Yuechao Lu
2015.03.18

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
__global__ void Maxscan(float *g_odata, float *g_idata, int dataSize);

int main()
{
	float *hostA, *hostB, *deviceA, *deviceB;
	unsigned int test_size = 8;
	cudaError_t hostMemStatus = cudaMallocHost((void**)&hostA, test_size * sizeof(float));
	if (hostMemStatus != cudaSuccess) { cout << "Error allocating pinned host memory" << endl; }

	hostMemStatus = cudaMallocHost((void**)&hostB, test_size * sizeof(float));
	if (hostMemStatus != cudaSuccess) { cout << "Error allocating pinned host memory" << endl; }

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

	dim3 dimGrid(1, 1);
	dim3 dimBlock(test_size, 1);
	Maxscan << < dimGrid, dimBlock, test_size >> >(deviceB, deviceA, test_size);


	/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from host memory A to device memory A ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
	cudaMemcpy(hostB, deviceB, test_size * sizeof(float), cudaMemcpyDeviceToHost);

	cout << "Output data:" << endl;
	for (int i = 0; i < test_size; i++){
		cout << hostB[i] << "	";
	}
	cout << endl;

	cudaFreeHost(hostA);
	cudaFreeHost(hostB);
	cudaFree(deviceA);
	cudaFree(deviceB);


	/*stop the console to see the results */
	cout << endl << "The end."; char str1[1];	scanf("%s", &str1);
	return 0;
}

void randomInit(float *data, int size){
	srand(2015);
	for (int i = 0; i < size; ++i)
	{
		data[i] = rand() / (float)RAND_MAX;
	}
}
/*This is also a exclusive scan */
__global__ void Maxscan(float *g_odata, float *g_idata, int dataSize){
	extern __shared__ float temp[];
	int tx = threadIdx.x;
	int offset = 1;
	temp[2 * tx] = g_idata[2 * tx];			/* load input into shared memory */
	temp[2 * tx + 1] = g_idata[2 * tx + 1];
	for (int i = dataSize >> 1; i > 0; i >>= 1){	/* build sum in place up the tree */
		__syncthreads();
		if (tx < i){
			int ai = offset * (2 * tx + 1) - 1;
			int bi = offset * (2 * tx + 2) - 1;
			if (temp[bi] < temp[ai]){
				temp[bi] = temp[ai];
			}
		}
		offset *= 2;
	}
	//set the last item to 0, cos 0 is the identity element of ADD operation
	if (tx == dataSize - 1)	{ temp[tx] = 0; }

	for (int i = 1; i < dataSize; i *= 2){	// traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if (tx < i){
			int ai = offset * (2 * tx + 1) - 1;
			int bi = offset * (2 * tx + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			if (temp[bi] < t){
				temp[bi] = t;
			}
		}
	}
	__syncthreads();

	g_odata[2 * tx] = temp[2 * tx];
	g_odata[2 * tx + 1] = temp[2 * tx + 1];
}
