#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

#define N 760		// side of matrix containing data
#define PDIM 768 	// padded dimensions to multiply of 32 (768 = 24 * 32)
#define DIV 6

/*
 * cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height)
 * The pitch returned is multiply of 512 (byte), which means the width is padded into multiply of 512.
 * the address of a entry in the allocated 2D memory is calcuated as
 * T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
 * I think BaseAddress = devPtr
 *
 *
 **/



//load element from da to db to verify correct memcopy
__global__ void kernel(float * dev_A, float * dev_B){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid % PDIM < N){
        int row = blockIdx.x / DIV, col = blockIdx.x % DIV;
        dev_B[row * N + col * blockDim.x + threadIdx.x] = dev_A[tid];
    }
}

void verify(float * A, float * B, int size);

int main(int argc, char * argv[]){

    float *host_A, *dev_A, *host_B, *dev_B;

    host_A = (float *)malloc(sizeof(float) * N * N);
    host_B = (float *)malloc(sizeof(float) * N * N);

    for (int i = 0; i < N * N; i++){
        host_A[i] = i;
    }
    size_t pitch;
    // malloc 2D memory dev_A
    checkCudaErrors(cudaMallocPitch(&dev_A, &pitch, sizeof(float) * N, N)); //
    printf("memory pitch (or number of column of the 2D memory) = %lu \n", pitch);

    // malloc 1D memory dev_B
    checkCudaErrors(cudaMalloc(&dev_B, sizeof(float) * N * N));

    // copy host A to 2D memory dev_A
    checkCudaErrors(cudaMemcpy2D(dev_A, pitch, host_A, sizeof(float) * N, sizeof(float) * N, N, cudaMemcpyHostToDevice));

    int threadsperblock = 128;
    int blockspergrid = PDIM * PDIM / threadsperblock;

    kernel <<< blockspergrid, threadsperblock >>> (dev_A, dev_B);


    checkCudaErrors(cudaMemcpy(host_B, dev_B, sizeof(float) * N * N, cudaMemcpyDeviceToHost));
    //cudaMemcpy2D(B,N,dB,N,N,N,cudaMemcpyDeviceToHost);

    verify(host_A, host_B, N * N);

    // test 2
    // set 2D memory dev_A to all 1, copyt to host and check the result
    int memSetValue = 0; // Value to set for each byte of specified memory, so memSetValue can only be 0 for float
    //checkCudaErrors(cudaMemset2D(dev_A, pitch, 0, sizeof(float) * N, N));
    checkCudaErrors(cudaMemset2D(dev_A, pitch, memSetValue, sizeof(float) * N, N));
    checkCudaErrors(cudaMemcpy2D(host_A,  sizeof(float) * N, dev_A, pitch, sizeof(float) * N, N, cudaMemcpyDeviceToHost));

    unsigned long long sum = 0;
    for (int i = 0; i < N * N; i++){
    	sum += host_A[i];
    }
    printf("sum = %lu \n", sum);
    if(sum == N * N * memSetValue){
    	printf("Memset2D works correctly\n");
    }else{
    	printf("Memset2D ERROR\n");
    }


    free(host_A);
    free(host_B);
    checkCudaErrors(cudaFree(dev_A));
    checkCudaErrors(cudaFree(dev_B));
}

void init(float * array, int size)
{

}

void verify(float * A, float * B, int size)
{
    for (int i = 0; i < size; i++){
        assert(A[i] == B[i]);
    }
    printf("Correct!");
}
