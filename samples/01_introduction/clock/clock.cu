#include <cuda_runtime.h>

#include <iostream>

// 找到长度为NUM_THREADS的数组的最小值
// 一共找NUM_BLOCKS次，求平均耗时
__global__ static void timeReduction(const float* input, float* output,
                clock_t* timer) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) {
        timer[bid] = clock();
    }
    
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // find minimum
    for (int i = blockDim.x; i > 0; i >>= 1) {
        __syncthreads();
        if (tid < i) {
            float f0 = shared[tid];
            float f1 = shared[tid + i];

            if (f0 < f1) {
                shared[tid] = f0;
            }
        }
    }

    if (tid == 0) {
        output[bid] = shared[0];
    }
    __syncthreads();
    
    if (tid == 0) {
        timer[bid + gridDim.x] = clock();
    }
}

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main() {
    cudaDeviceProp dev = cudaDeviceProp();
    cudaGetDeviceProperties(&dev, 0);
    printf("Device name: %s\n", dev.name);
    printf("Clock rate: %d\n", dev.clockRate);
    printf("Compute capability: %d.%d\n", dev.major, dev.minor);
    printf("Max grid size: %d, %d, %d\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
    printf("Max threads per block: %d\n", dev.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", dev.maxThreadsPerMultiProcessor);
    printf("Max threads dim: %d, %d, %d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
    printf("Max shared memory per block: %d\n", dev.sharedMemPerBlock);
    printf("Max shared memory per multiprocessor: %d\n", dev.sharedMemPerMultiprocessor);
    printf("Max registers per block: %d\n", dev.regsPerBlock);
    printf("Max registers per multiprocessor: %d\n", dev.regsPerMultiprocessor);
    printf("MultiProcessor count: %d\n", dev.multiProcessorCount);
    printf("Max blocks per multiprocessor: %d\n", dev.maxBlocksPerMultiProcessor);
    printf("Warp size: %d\n", dev.warpSize);
    printf("totalGlobalMem: %Ld\n", dev.totalGlobalMem);

    printf("size of (long double) is %d bytes\n", sizeof(long double));

    float* d_input = nullptr;
    float* d_output = nullptr;
    clock_t* d_timer = nullptr;

    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; ++i) {
        input[i] = (float)i;
    }
    
    cudaMalloc((void**)&d_input, sizeof(float) * NUM_THREADS * 2);
    cudaMalloc((void**)&d_output, sizeof(float) * NUM_BLOCKS);
    cudaMalloc((void**)&d_timer, sizeof(clock_t) * NUM_BLOCKS * 2);

    cudaMemcpy(d_input, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice);

    timeReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * NUM_THREADS * 2>>>(d_input, d_output, d_timer);

    cudaMemcpy(timer, d_timer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_timer);

    long double avgElapsedClocks = 0;
    // timer[i]记录开始时间，timer[i + NUM_BLOCKS]记录结束时间
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }
    avgElapsedClocks /= NUM_BLOCKS;
    printf("Average elapsed clocks: %Lf\n", avgElapsedClocks);

    return 0;
}