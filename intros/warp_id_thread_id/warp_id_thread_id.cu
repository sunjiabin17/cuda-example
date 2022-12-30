#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_details_of_warps() {
    int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / 32;
    
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    printf("gid: %d, warp_id: %d, gbid: %d\n", gid, warp_id, gbid);
}

int main() {
    dim3 block_size(42);
    dim3 grid_size(2,2);

    print_details_of_warps<<<grid_size, block_size>>>();
    
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}