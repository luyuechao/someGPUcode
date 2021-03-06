/*
 test host to device transfer time
 Yuechao Lu
 2015.03.09
 */

// System includes
#include <iostream>
#include <fstream>

using namespace std;

// CUDA runtime

// error checker
#define CHECK_CUDA(call)                                                       \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);            \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


// Helper functions and utilities to work with CUDA
void randomInit(float *data, unsigned long long size);

int main()
{
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓setup output file in C sytle↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    ofstream outputfile;
    outputfile.open("outputData.csv");
    if (outputfile.fail())	{	cout << "fail to open file" << endl;		}
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑setup output file in C sytle↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
    
    float *hostA,*hostB,*deviceA,*deviceB;

    float timer1 =0.0, timer2 = 0.0, timer3 = 0.0, timer4 = 0.0, timer5 = 0.0;
    
    for (unsigned long long test_size = 1024; test_size <= 1024 * 1024 * 1024; test_size = 4 * test_size){
        timer1 =0.0; timer2 = 0.0; timer3 = 0.0; timer4 = 0.0; timer5 = 0.0;

        unsigned long long  testBytes = test_size * sizeof(float);
        cout << endl ;
        if (testBytes < 1024 * 1024){
            cout << testBytes / 1024 << " KB";
        }else{
            cout << testBytes / (1024*1024) << " MB";
        }
        cout << " Transfer bandwitdth(GB/s)" << endl;
        //cout <<"Allocating host memory A as pinned memory..."<<endl;
        CHECK_CUDA( cudaMallocHost((void**)&hostA, testBytes) );

        
        //cout <<"Allocating host memory B as pageable memory..."<<endl;
        hostB = (float *)malloc(testBytes);
        
        //cout <<"Allocating device memory..."<<endl;
        CHECK_CUDA( cudaMalloc((void **)&deviceA, testBytes) );
        CHECK_CUDA( cudaMalloc((void **)&deviceB, testBytes) );
        
        // initialize host memory
        randomInit(hostA, test_size);
        randomInit(hostB, test_size);
        
        //printf("Creating timer...\n");
        cudaEvent_t start, stop;
        CHECK_CUDA( cudaEventCreate(&start) );
        CHECK_CUDA( cudaEventCreate(&stop) );

        
        outputfile << testBytes << ",";
        
        /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from host memory A to device memory A ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
        CHECK_CUDA( cudaEventRecord(start, 0) );
        CHECK_CUDA( cudaMemcpy(deviceA, hostA, testBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaDeviceSynchronize() );	/*tell CPU to wait until GPU finish job*/
        
        CHECK_CUDA( cudaEventRecord(stop, 0) );
        CHECK_CUDA( cudaEventSynchronize(stop) );//blocks CPU execution until the specified event is recorded
        CHECK_CUDA( cudaEventElapsedTime(&timer1, start, stop) );//record milliseconds between 2 events
        outputfile << timer1 << ",";
        cout << "	Host Memory A (Pinned) to Device  memory A:	"
        << ((float)testBytes) * (1.0e-6) / timer1 << endl;
        /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑Copy from host memory A to device memory A ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
        
        /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from host memory B to device memory B ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
        CHECK_CUDA( cudaEventRecord(start, 0) );
        CHECK_CUDA( cudaMemcpy(deviceB, hostB, testBytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaDeviceSynchronize() );	/*tell CPU to wait until GPU finish job*/
        CHECK_CUDA( cudaEventRecord(stop, 0) );
        CHECK_CUDA( cudaEventSynchronize(stop) );
        CHECK_CUDA( cudaEventElapsedTime(&timer2, start, stop) );
        outputfile << timer2 << ",";
        cout << "	Host Memory B (pageable) to Device  memory B:	"
        << ((float)testBytes) * (1.0e-6) / timer2 << endl;
        /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑Copy from host memory B to device memory B  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
        
        /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from device memory A to device memory B ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
        CHECK_CUDA( cudaEventRecord(start, 0) );
        CHECK_CUDA( cudaMemcpy(deviceB, deviceA, testBytes, cudaMemcpyDeviceToDevice) );
        CHECK_CUDA( cudaDeviceSynchronize() );	/*tell CPU to wait until GPU finish job*/
        CHECK_CUDA( cudaEventRecord(stop, 0) );
        CHECK_CUDA( cudaEventSynchronize(stop) );
        CHECK_CUDA( cudaEventElapsedTime(&timer3, start, stop) );
        outputfile << timer3 << ",";
        cout << "	Device Memory A to Device memory B:		"
        << ((float)testBytes) * (1.0e-6) / timer3 << endl;
        /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑Copy from device memory A to device memory B ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
        
        /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from device memory B to host memory A ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
        CHECK_CUDA( cudaEventRecord(start, 0) );
        CHECK_CUDA( cudaMemcpy(hostA, deviceB, testBytes, cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaDeviceSynchronize() );	/*tell CPU to wait until GPU finish job*/
        CHECK_CUDA( cudaEventRecord(stop, 0) );
        CHECK_CUDA( cudaEventSynchronize(stop) );
        CHECK_CUDA( cudaEventElapsedTime(&timer4, start, stop) );
        outputfile << timer4 << ",";
        cout << "	Device Memory B to Host Memory A (Pinned):	"
        << ((float)testBytes) * (1.0e-6) / timer4 << endl;
        /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑Copy from device memory B to host memory A ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
        
        /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓Copy from device memeory B to host memeory B ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
        CHECK_CUDA( cudaEventRecord(start, 0) );
        CHECK_CUDA( cudaMemcpy(hostB, deviceB, testBytes, cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaDeviceSynchronize() );	/*tell CPU to wait until GPU finish job*/
        CHECK_CUDA( cudaEventRecord(stop, 0) );
        CHECK_CUDA( cudaEventSynchronize(stop) );
        CHECK_CUDA( cudaEventElapsedTime(&timer5, start, stop) );
        outputfile << timer5 << endl;
        cout << "	Device Memory B to Host Memory B (pageable):	"
        << ((float)testBytes) * (1.0e-6) / timer5 << endl;
        /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑Copy from device memeory B to host memeory B ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
        
        CHECK_CUDA( cudaEventDestroy(start) );
        CHECK_CUDA( cudaEventDestroy(stop) );
        CHECK_CUDA( cudaFreeHost(hostA) );
        free(hostB);
        CHECK_CUDA( cudaFree(deviceA) );
        CHECK_CUDA( cudaFree(deviceB) );
    }
    
    outputfile.close();
    /*stop the console to see the results */
    cout << "Test finished." << endl;
    return 0;
}

void randomInit(float *data, unsigned long long size){
    srand(2015);
    for (unsigned long long i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
