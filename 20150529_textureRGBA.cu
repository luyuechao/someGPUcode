
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PIC_WIDTH 4
#define PIC_HEIGHT 4	
static const int picSize = 16;
#define PIC_LAYER_NUM 4



__global__ void readTexture(float *redTex, float *greenTex, float *blueTex, float *alphaTex, float *output, cudaTextureObject_t texRGBA){
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	float4 rgba;

	redTex[Y* PIC_WIDTH + X]	= tex2DLayered<float>(texRGBA, X + 0.4, Y + 0.3, 0);
	greenTex[Y* PIC_WIDTH + X]	= tex2DLayered<float>(texRGBA, X + 0.4, Y + 0.3, 1);
	blueTex[Y* PIC_WIDTH + X]	= tex2DLayered<float>(texRGBA, X + 0.4, Y + 0.3, 2);
	alphaTex[Y* PIC_WIDTH + X]	= tex2DLayered<float>(texRGBA, X + 0.4, Y + 0.3, 3);

	output[Y* PIC_WIDTH * PIC_LAYER_NUM + X * PIC_LAYER_NUM] = redTex[Y* PIC_WIDTH + X];
	output[Y* PIC_WIDTH * PIC_LAYER_NUM + X * PIC_LAYER_NUM + 1] = greenTex[Y* PIC_WIDTH + X];
	output[Y* PIC_WIDTH * PIC_LAYER_NUM + X * PIC_LAYER_NUM + 2] = blueTex[Y* PIC_WIDTH + X];
	output[Y* PIC_WIDTH * PIC_LAYER_NUM + X * PIC_LAYER_NUM + 3] = alphaTex[Y* PIC_WIDTH + X];
}


int main()
{
	float *texRed, *texGreen, *texBlue, *texAlpha;

	texRed = (float *)malloc(sizeof(float) * picSize);
	texGreen = (float *)malloc(sizeof(float) * picSize);
	texBlue = (float *)malloc(sizeof(float) * picSize);
	texAlpha = (float *)malloc(sizeof(float) * picSize);

	float *host_RGBA;
	host_RGBA = (float *)malloc(sizeof(float) * picSize * PIC_LAYER_NUM);
	float r[16] = { 0.28, 0.43, 0.34, 0.45, 0.35, 0.98, 0.14, 0.32, 0.34, 0.33, 0.63, 0.20, 0.25, 0.29, 0.42, 0.10 };
	float g[16] = { 2.35, 2.05, 2.56, 2.45, 2.43, 2.19, 2.41, 2.02, 2.53, 2.14, 2.23, 2.09, 2.81, 2.53, 2.10, 2.52 };
	float b[16] = { 11.35, 11.12, 11.56, 11.45, 11.43, 11.19, 11.41, 11.02, 11.53, 11.14, 11.23, 11.09, 11.81, 11.53, 11.10, 11.52 };
	float a[16] = { 7.35, 7.15, 7.56, 7.45, 7.43, 7.19, 7.41, 7.02, 7.53, 7.14, 7.23, 7.09, 7.81, 7.53, 7.10, 7.52 };

	for (int i = 0; i < picSize; i++){
		host_RGBA[i] = r[i];
		host_RGBA[i + picSize] = g[i];
		host_RGBA[i + picSize * 2] = b[i];
		host_RGBA[i + picSize * 3] = a[i];

	}

	memcpy(host_RGBA, r, picSize);
	memcpy(&host_RGBA[picSize], g, picSize);
	memcpy(&host_RGBA[picSize * 2], b, picSize);
	memcpy(&host_RGBA[picSize * 3], a, picSize);


	printf("RGBA =\n");
	for (int i = 0; i < PIC_HEIGHT; i++){
		for (int j = 0; j < PIC_WIDTH * PIC_LAYER_NUM; j++){
			printf("%.2f	", host_RGBA[i * PIC_WIDTH + j]);
		}
		printf("\n");
	}

	cudaArray_t tex_buf;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaExtent extent;
	extent.width = PIC_WIDTH;
	extent.height = PIC_HEIGHT;
	extent.depth = PIC_LAYER_NUM;
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
	texDescr.filterMode = cudaFilterModeLinear;   // cudaFilterModePoint is not filtered

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;  /**< Read texture as specified element type */
	checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

	/*---------------- for copy data --------------------- */
	cudaMemcpy3DParms myparms = { 0 };
	myparms.srcPos = make_cudaPos(0, 0, 0);
	myparms.dstPos = make_cudaPos(0, 0, 0);
	myparms.srcPtr = make_cudaPitchedPtr(host_RGBA, PIC_WIDTH * sizeof(float), PIC_WIDTH, PIC_HEIGHT);
	myparms.dstArray = tex_buf;
	myparms.extent = make_cudaExtent(PIC_WIDTH, PIC_HEIGHT, PIC_LAYER_NUM);
	myparms.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&myparms));

	float *deviceMem[PIC_LAYER_NUM];
	for (int i = 0; i < PIC_LAYER_NUM; i++){
		checkCudaErrors(cudaMalloc((void**)&deviceMem[i], sizeof(float) * picSize));
	}

	float *deviceRGBA;
	checkCudaErrors(cudaMalloc((void**)&deviceRGBA, sizeof(float) * picSize * PIC_LAYER_NUM));

	dim3 grid_tex(1, 1);
	dim3 thread_tex(4, 4);

	readTexture << < grid_tex, thread_tex >> > (deviceMem[0], deviceMem[1], deviceMem[2], deviceMem[3], deviceRGBA, texInput);

	cudaMemcpy(texRed, deviceMem[0], sizeof(float) * picSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(texGreen, deviceMem[1], sizeof(float) * picSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(texBlue, deviceMem[2], sizeof(float) * picSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(texAlpha, deviceMem[3], sizeof(float) * picSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_RGBA, deviceRGBA, sizeof(float) * picSize * PIC_LAYER_NUM, cudaMemcpyDeviceToHost);

	printf("Red =\n");
	for (int i = 0; i < picSize; i++){
		printf("%.2f	", texRed[i]);
	}
	printf("\n");

	printf("Green =\n");
	for (int i = 0; i < picSize; i++){
		printf("%.2f	", texGreen[i]);
	}
	printf("\n");

	printf("Blue =\n");
	for (int i = 0; i < picSize; i++){
		printf("%.2f	", texBlue[i]);
	}
	printf("\n");

	printf("Alpha =\n");
	for (int i = 0; i < picSize; i++){
		printf("%.2f	", texAlpha[i]);
	}
	printf("\n");

	printf("RGBA =\n");
	for (int i = 0; i < PIC_HEIGHT; i++){
		for (int j = 0; j < PIC_WIDTH * PIC_LAYER_NUM; j++){
			printf("%.4f	", host_RGBA[i * PIC_WIDTH + j]);
		}
		printf("\n");
	}

	return 0;
}
