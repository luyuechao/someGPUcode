/*
 https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 
*/

#include <stdio.h>
#include <time.h>
#include <cublas_v2.h>


#include "gpuErrorCheck.h"

#define WARP_SIZE 32

#if __CUDA_ARCH__ < 500
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif




__host__ __device__
static inline uint64_t roundup_to_32X(const int x){
    return ((x + 32 - 1) / 32) * 32;
}

__inline__ __device__ void warpReduce(float &val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        val += __shfl_down(val, offset);
        // val += __shfl_xor(val, i); //butterfly style TODO: what's the difference?
    }
}

__inline__ __device__ void blockReduce(float &val){
    
    static __shared__ float sharedMem[WARP_SIZE];
    
    int lane = threadIdx.x % warpSize;
    
    int wid = threadIdx.x / warpSize; // warp ID
    
    warpReduce(val);
    
    if(lane == 0){ // the first thread in every warp write to shared memory
        sharedMem[wid] =val;
    }
    
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / warpSize) ? sharedMem[lane] : 0;
    
    if(wid == 0){
        warpReduce(val);
    }
}

// because reduce between different thread block requires using global memory to share
// data and sychronize between thread blocks, so use single tb to scan the data is
// faster (maybe).
__global__ void reduceKernel(float *d_sum, const float *d_data, const int lda){

    int tx = threadIdx.x;
    
    float sum = 0.0;
    
    for(int i = tx; i < lda; i += blockDim.x)
    {
        sum += d_data[i];
    }

    blockReduce(sum);
    
    //printf("tx = %d, sum = %d\n", tx, sum);
    
    if(tx == 0){
        d_sum[tx] = sum;
    }
    
}


bool reduceTest(const unsigned long long m, cublasHandle_t &cublasH){
    // host data
    float *h_data = (float *)malloc(sizeof(float) * m );
    float h_sum = 0;
    
    for(int i = 0; i < m; i++){
        h_data[i] = (float)i;
    }
    
    
    const double sum_gt = (double(m) - 1.0) * (double(m) / 2.0);
    

    
    // round up to multiple of 32 for warp shuffle see:
    int lda = roundup_to_32X(m);
    
    //printf("lda = %d.\n", lda);
    
    // device data
    float *d_data, *d_sum;
    CHECK_CUDA( cudaMalloc( (void **)&d_data, sizeof(float) * lda ) );
    CHECK_CUDA( cudaMemset(d_data, 0, sizeof(float) * lda ) );
    CHECK_CUDA( cudaMalloc( (void **)&d_sum, sizeof(float) ) );
    
    CHECK_CUDA( cudaMemcpy(d_data, h_data, sizeof(float) * m, cudaMemcpyHostToDevice) );
    
    uint64_t tick = clock();
    reduceKernel <<< 1, MAX_THREADS >>> (d_sum, d_data, lda);
    
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA( cudaGetLastError() );
    
    
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    uint64_t tock = clock();
    
    double time = (tock - tick); //µs
    
    float diff = h_sum - sum_gt;
//    if( diff > 1e-1 || diff < -1e-1){
//        
//        double ftemp = (double(m) - 1.0);
//        printf("m - 1 = %.2f\n", ftemp);
//        ftemp *= double(m);
//        printf("Here comes the error: m = %d\n", m);
//        printf("(m - 1) * m = %.2f\n", ftemp);
//        printf("(m - 1) * m / 2.0 = %.2f\n", ftemp / 2.0);
//
//        printf("data size = %.3f GB.\n", (double)m * sizeof(float) /(1024.0 * 1024.0 * 1024.0) );
//        
//        printf("correct sum = %.2f.\n", sum_gt );
//        printf("sum by handcoded cuda: %.2f, time = %.2f µs.\n", h_sum, time);
//        return true;
//    }
    
    
    tick = clock();
    // cublasSasum calculate the l1 norm of a vector
    CHECK_CUBLAS( cublasSasum(cublasH, m, d_data, 1, &h_sum) );
    tock = clock();
    
    time = (tock - tick);
    
    diff = h_sum - sum_gt;
    if( diff > 1e-1 || diff < -1e-1){
        
        printf("Here comes the error. m = %d\n", m);
        printf("data size = %.3f GB.\n", (double)m * sizeof(float) /(1024.0 * 1024.0 * 1024.0) );
        
        printf("correct sum = %.2f.\n", sum_gt );
        printf("sum by cuBLAS: %.2f, time = %.2f µs.\n", h_sum, time);
        return true;
    }
    // clean ups

    free( h_data );
    CHECK_CUDA( cudaFree(d_data) );
    CHECK_CUDA( cudaFree(d_sum)  );
    
    return false;
}

int main(int argc, char **argv)
{
    float tick = clock();
    cublasHandle_t cublasH = NULL;
    CHECK_CUBLAS( cublasCreate(&cublasH) );
    float tock = clock();
    bool result = false;
    for(unsigned long long m = 1000; m < 1024 * 1024 * 1024; m++){
        result = reduceTest(m, cublasH);
        if(result) break;
    }
    
    CHECK_CUBLAS( cublasDestroy(cublasH) );
    CHECK_CUDA( cudaDeviceReset() );
    
    return 0;
}

