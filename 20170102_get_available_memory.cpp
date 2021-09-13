cudaDeviceProp deviceProp;

CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));

size_t gpuGlobalMem = deviceProp.totalGlobalMem;

fprintf(stderr, "GPU global memory = %zu Bytes\n", gpuGlobalMem);

size_t freeMem, totalMem;

CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
double GB = double(1024 * 1024 * 1024);
 fprintf(stderr, "Free = %0.1f GB, Total = %0.1f GB\n", double(freeMem) / GB, double(totalMem) / GB);
