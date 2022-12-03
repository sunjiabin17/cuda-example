#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_idx_calculation(int* input) {
    int idx = threadIdx.x;
    int unique_idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("threadIdx: %d, unique_idx: %d, value: %d\n", idx, unique_idx, input[unique_idx]);
}

__global__ void unique_idx_calculation_2d(int* input) {
    // grid 2维, block 1维
    // index = row offset + block offset + thread offset
    // row offset = blockIdx.y * blockDim.x * gridDim.x
    // block offset = blockIdx.x * blockDim.x
    // thread offset = threadIdx.x

    // int rowoffset = blockIdx.y * blockDim.x * gridDim.x;
    // int blockoffset = blockIdx.x * blockDim.x;
    // int unique_idx = rowoffset + blockoffset + threadIdx.x;
    // printf("rowoffset: %d, blockoffset: %d, unique_idx: %d, value: %d\n", rowoffset, blockoffset, unique_idx, input[unique_idx]);


    // grid 2维, block 2维
    int thread_offset = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads_in_a_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_in_a_block;
    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
    int row_offset = blockIdx.y * num_threads_in_a_row;
    
    int unique_idx = thread_offset + block_offset + row_offset;

    // 方案2
    // int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
    // int unique_idx = block_offset * num_threads_in_a_block + thread_offset;

    printf("thread_offset: %d, block_offset: %d, row_offset: %d, unique_idx: %d, value: %d\n", thread_offset, block_offset, row_offset, unique_idx, input[unique_idx]);

}

__global__ void unique_idx_calc_3d(int* input) {

    int thread_offset = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    int num_threads_in_a_block = blockDim.x * blockDim.y * blockDim.z;
    
    int block_offset = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

    int idx = thread_offset + block_offset * num_threads_in_a_block;

    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, idx: %d, value: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, idx, input[idx]);
}


int main() {
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_input[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    for (int i = 0; i < array_size; ++i) {
        printf("%d ", h_input[i]);
    }
    printf("\n\n");
    int *d_input;
    cudaMalloc((void**)&d_input, array_byte_size);
    cudaMemcpy(d_input, h_input, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2, 2);
    dim3 grid(2, 2);

    unique_idx_calculation_2d <<< grid, block >>> (d_input);
    cudaDeviceSynchronize();

    // cuda device reset
    cudaDeviceReset();

    return 0;
}