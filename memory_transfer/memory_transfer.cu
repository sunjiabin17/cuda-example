#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// __global__ void unique_idx_calculation_2d(int* input) {
//     // grid 2维, block 1维
//     // index = row offset + block offset + thread offset
//     // row offset = blockIdx.y * blockDim.x * gridDim.x
//     // block offset = blockIdx.x * blockDim.x
//     // thread offset = threadIdx.x

//     // int rowoffset = blockIdx.y * blockDim.x * gridDim.x;
//     // int blockoffset = blockIdx.x * blockDim.x;
//     // int unique_idx = rowoffset + blockoffset + threadIdx.x;
//     // printf("rowoffset: %d, blockoffset: %d, unique_idx: %d, value: %d\n", rowoffset, blockoffset, unique_idx, input[unique_idx]);


//     // grid 2维, block 2维
//     int thread_offset = threadIdx.y * blockDim.x + threadIdx.x;
//     int num_threads_in_a_block = blockDim.x * blockDim.y;
//     int block_offset = blockIdx.x * num_threads_in_a_block;

//     int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
//     int row_offset = blockIdx.y * num_threads_in_a_row;

//     int unique_idx = thread_offset + block_offset + row_offset;
//     printf("thread_offset: %d, block_offset: %d, row_offset: %d, unique_idx: %d, value: %d\n", thread_offset, block_offset, row_offset, unique_idx, input[unique_idx]);

// }

__global__ void memory_transfer_test(int* input, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 用size控制执行的thread数，大于size的thread不执行
    if (idx < size) {
        printf("idx: %d, value: %d\n", idx, input[idx]);
    }
}

int main() {
    int size = 150;
    int byte_size = size * sizeof(int);

    int* h_input = (int*)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; ++i) {
        h_input[i] = (int)(rand() & 0xff);
    }

    int* d_input;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(5);

    memory_transfer_test <<< grid, block >>> (d_input, size);
    cudaDeviceSynchronize();

    // cuda device reset
    cudaDeviceReset();

    return 0;
}