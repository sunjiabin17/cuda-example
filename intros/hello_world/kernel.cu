#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// cuda hello world
__global__ void helloFromGPU()
{
    printf("Hello World from GPU! \n");
}

int main() {
    // print hello world from cpu
    printf("Hello World from CPU!");

    // print hello world from gpu
    dim3 block(4, 1, 1);
    dim3 grid(8, 1, 1);
    helloFromGPU <<<grid, block>>>();
    cudaDeviceSynchronize();

    // cuda device reset
    cudaDeviceReset();

    return 0;
}