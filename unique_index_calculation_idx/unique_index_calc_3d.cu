#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_idx_calc_3d(int* input) {

    int thread_offset = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    int num_threads_in_a_block = blockDim.x * blockDim.y * blockDim.z;
    
    int block_offset = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

    int idx = thread_offset + block_offset * num_threads_in_a_block;

    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, idx: %d, value: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, idx, input[idx]);
}

int main() {
    int size = 64;
    int h_input[size];
    int byte_size = size * sizeof(int);
    for (int i = 0; i < size; ++i) {
        h_input[i] = i+1;
    }
    int* d_input;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);

    unique_idx_calc_3d <<<grid, block>>> (d_input);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}