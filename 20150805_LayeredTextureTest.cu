/*
This program is used test how the texture memory is arrange on the global memeory
so as to find the fastest access pattern.

*/

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>


#define PIC_WIDTH 1024
#define PIC_HEIGHT 1024
static const int picSize = PIC_WIDTH * PIC_HEIGHT;
#define picLayerNum 64

struct texObjtStrut { cudaTextureObject_t texAry[picLayerNum]; };
/*pattern 1 */
__global__ void readTextureLayerByLayer(float *output, cudaTextureObject_t LayerTex){

	float ftemp = 0.0f;
#pragma unroll
	for (int layer = 0; layer < picLayerNum; layer++){
		for (int row = 0; row < PIC_HEIGHT; row++) {
			for (int col = 0; col < PIC_WIDTH; col++) {
				ftemp += tex2DLayered<float>(LayerTex, row, col, layer);
			}
		}
	}
	*output = ftemp;
}
/*pattern 2*/
__global__ void readTextureDotByDot(float *output, cudaTextureObject_t LayerTex){

	float ftemp = 0.0f;

	for (int row = 0; row < PIC_HEIGHT; row++) {
		for (int col = 0; col < PIC_WIDTH; col++) {
#pragma unroll
			for (int layer = 0; layer < picLayerNum; layer++){
				ftemp += tex2DLayered<float>(LayerTex, row, col, layer);
			}
		}
	}
	*output = ftemp;
}
/*pattern 3*/
__global__ void readTextureObjtByObjt(float *output, texObjtStrut texObjtSet){

	float ftemp = 0.0f;
#pragma unroll
	for (int objt = 0; objt < picLayerNum; objt++){
		for (int row = 0; row < PIC_HEIGHT; row++) {
			for (int col = 0; col < PIC_WIDTH; col++) {
				ftemp += tex2D<float>(texObjtSet.texAry[objt], row, col);
			}
		}
	}
	*output = ftemp;
}

/*pattern4 put different pic in diferent tex obj and
access the tex by the loop sequence of row->col->tex*/
__global__ void pattern4(float *output, texObjtStrut texObjtSet){

	float ftemp = 0.0f;

	for (int row = 0; row < PIC_HEIGHT; row++) {
		for (int col = 0; col < PIC_WIDTH; col++) {
#pragma unroll
			for (int objt = 0; objt < picLayerNum; objt++){
				ftemp += tex2D<float>(texObjtSet.texAry[objt], row, col);
			}
		}
	}
	*output = ftemp;
}
/*pattern 5 is all the same with pattern 3 except from the unroll setup*/
__global__ void pattern5(float *output, texObjtStrut texObjtSet){

	float ftemp = 0.0f;

	for (int objt = 0; objt < picLayerNum; objt++){
		for (int row = 0; row < PIC_HEIGHT; row++) {
#pragma unroll 100
			for (int col = 0; col < PIC_WIDTH; col++) {
				ftemp += tex2D<float>(texObjtSet.texAry[objt], row, col);
			}
		}
	}
	*output = ftemp;
}

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("kernelExecTimeoutEnabled = %d\n",prop.kernelExecTimeoutEnabled);
	srand(2015);

	float *pictureSET;
	checkCudaErrors(cudaHostAlloc((void**)&pictureSET, sizeof(float) * picLayerNum * PIC_WIDTH * PIC_HEIGHT, cudaHostAllocDefault));
	for (int i = 0; i < picLayerNum * PIC_WIDTH * PIC_HEIGHT; i++){
		pictureSET[i] = rand() / (float)RAND_MAX;
	}

	float hostmem = 0.0f;


	cudaArray_t tex_buf;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaExtent extent;
	extent.width = PIC_WIDTH;
	extent.height = PIC_HEIGHT;
	extent.depth = picLayerNum;
	checkCudaErrors(cudaMalloc3DArray(&tex_buf, &desc, extent, cudaArrayLayered));

	// generate texture object for reading
	cudaTextureObject_t         texInput;
	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = tex_buf;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 0;		//Indicates whether texture reads are normalized or not
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;  /**< Read texture as specified element type */
	checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

	/*---------------- for copy data --------------------- */
	cudaMemcpy3DParms myparms = { 0 };
	myparms.srcPos = make_cudaPos(0, 0, 0);
	myparms.dstPos = make_cudaPos(0, 0, 0);
	myparms.srcPtr = make_cudaPitchedPtr(pictureSET, PIC_WIDTH * sizeof(float), PIC_WIDTH, PIC_HEIGHT);
	myparms.dstArray = tex_buf;
	myparms.extent = make_cudaExtent(PIC_WIDTH, PIC_HEIGHT, picLayerNum);
	myparms.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&myparms));


	float *deviceMem;
	checkCudaErrors(cudaMalloc((void**)&deviceMem, sizeof(float)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime;

	dim3 grid_tex(1, 1);
	dim3 thread_tex(1, 1);

	cudaEventRecord(start, 0);
	readTextureLayerByLayer << < grid_tex, thread_tex >> > (deviceMem, texInput);
	getLastCudaError("CUDA kernel failed\n");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time used layerbylayer (pattern 1) = %0.0f\n", elapsedTime);
	checkCudaErrors(cudaMemcpy(&hostmem, deviceMem, sizeof(float), cudaMemcpyDeviceToHost));
	printf("result = %f\n", hostmem);
	memset(&hostmem, 0, sizeof(hostmem));

	cudaEventRecord(start, 0);
	readTextureDotByDot << < grid_tex, thread_tex >> > (deviceMem, texInput);
	getLastCudaError("CUDA kernel failed\n");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time used dotbydot (pattern 2) = %0.0f\n", elapsedTime);
	checkCudaErrors(cudaMemcpy(&hostmem, deviceMem, sizeof(float), cudaMemcpyDeviceToHost));
	printf("result = %f\n", hostmem);
	memset(&hostmem, 0, sizeof(hostmem));
	
	checkCudaErrors(cudaFreeArray(tex_buf));
	checkCudaErrors(cudaDestroyTextureObject(texInput));

	float* tex_data[picLayerNum];
	size_t pitch[picLayerNum];

	cudaResourceDesc resDesc;
	cudaTextureDesc texDesc;
	texObjtStrut texContainer;

	for (int i = 0; i < picLayerNum; i++){
		cudaMallocPitch(&tex_data[i], &pitch[i], sizeof(float)*PIC_WIDTH, PIC_HEIGHT);
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = tex_data[i];
		resDesc.res.pitch2D.pitchInBytes = pitch[i];
		resDesc.res.pitch2D.width = PIC_WIDTH;
		resDesc.res.pitch2D.height = PIC_HEIGHT;
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		texDesc.addressMode[0] = cudaAddressModeClamp;// X axis
		texDesc.addressMode[1] = cudaAddressModeClamp;// Y axis
		texDesc.filterMode = cudaFilterModeLinear;
		cudaCreateTextureObject(&texContainer.texAry[i], &resDesc, &texDesc, NULL);
		checkCudaErrors(cudaMemcpy2D(tex_data[i], pitch[i], &pictureSET[i],
			sizeof(float) * PIC_WIDTH, sizeof(float) * PIC_WIDTH,
			PIC_HEIGHT, cudaMemcpyHostToDevice));
	}



	
	
	cudaEventRecord(start, 0);
	readTextureObjtByObjt << < grid_tex, thread_tex >> > (deviceMem, texContainer);	
	getLastCudaError("CUDA kernel failed\n");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time used objtbyobjt (pattern 3) = %0.0f\n", elapsedTime);
	checkCudaErrors(cudaMemcpy(&hostmem, deviceMem, sizeof(float), cudaMemcpyDeviceToHost));
	printf("result = %f\n", hostmem);
	memset(&hostmem, 0, sizeof(hostmem));

	cudaEventRecord(start, 0);
	pattern4 << < grid_tex, thread_tex >> > (deviceMem, texContainer);
	getLastCudaError("CUDA kernel failed\n");
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("time used pattern4  = %0.0f\n", elapsedTime);
	(cudaMemcpy(&hostmem, deviceMem, sizeof(float), cudaMemcpyDeviceToHost));
	printf("result = %f\n", hostmem);

	cudaEventRecord(start, 0);
	pattern5 << < grid_tex, thread_tex >> > (deviceMem, texContainer);
	getLastCudaError("CUDA kernel failed\n");
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("time used pattern 5  = %0.0f\n", elapsedTime);
	(cudaMemcpy(&hostmem, deviceMem, sizeof(float), cudaMemcpyDeviceToHost));
	printf("result = %f\n", hostmem);


	for (int i = 0; i < picLayerNum; i++){
		checkCudaErrors(cudaFree(tex_data[i]));//
		checkCudaErrors(cudaDestroyTextureObject(texContainer.texAry[i]));
	}

	checkCudaErrors(cudaFreeHost(pictureSET));
	cudaEventDestroy(start);	cudaEventDestroy(stop);
	cudaDeviceReset();
	return 0;
}
