#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void printThreadIdx()
{
    printf("ThreadIdx.x=%d, ThreadIdx.y=%d, ThreadIdx.z=%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void printBlockIdx()
{
    printf("BlockIdx.x=%d, BlockIdx.y=%d, BlockIdx.z=%d\n", blockIdx.x, blockIdx.y, blockIdx.z);
}

__global__ void printBlockDim()
{
    printf("BlockDim.x=%d, BlockDim.y=%d, BlockDim.z=%d\n", blockDim.x, blockDim.y, blockDim.z);
}

__global__ void printGridDim()
{
    printf("GridDim.x=%d, GridDim.y=%d, GridDim.z=%d\n", gridDim.x, gridDim.y, gridDim.z);
}


int main() {
    int nx = 16, ny = 16;
    dim3 block(8, 8);
    dim3 grid(nx / block.x, ny / block.y);
    printGridDim <<<grid, block>>> ();
    
    cudaDeviceSynchronize();

    // cuda device reset
    cudaDeviceReset();

    return 0;
}