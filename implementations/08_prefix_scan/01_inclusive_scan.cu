#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//compare arrays
void compare_arrays(int * a, int * b, int size) {
    for (int  i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different \n");
            // printf("%d - %d | %d \n", i, a[i], b[i]);
            return;
        }
    }
    printf("Arrays are same \n");
}

void inclusive_scan_cpu(int *input, int *output, int size) {
    output[0] = input[0];
    for (int i = 1; i < size; i++)
    {
        output[i] = output[i - 1] + input[i];
    }
}

__global__ void inclusive_scan_naive(int *input, int size) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < size) {
        for (int stride = 1; stride <= gid; stride *= 2) {
            input[gid] += input[gid - stride];
            __syncthreads();
        }
    }
}

__global__ void inclusive_scan_pre(int *input, int* aux, int size) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int shared_mem[];
    if (gid < size) {
        shared_mem[tid] = input[gid];
        __syncthreads();

        // reduce phrase
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int idx = (tid + 1) * stride * 2 - 1;
            if (idx < blockDim.x) {
                shared_mem[idx] += shared_mem[idx - stride];
            }
            __syncthreads();
        }

        // down sweep phrase
        for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
            int idx = (tid + 1) * stride * 2 - 1;
            if (idx + stride < blockDim.x) {
                shared_mem[idx + stride] += shared_mem[idx];
            }
            __syncthreads();
        }

        // write back to global memory
        input[gid] = shared_mem[tid];
        if (tid == 0) {
            aux[blockIdx.x] = shared_mem[blockDim.x - 1];
        }
    }
}

__global__ void inclusive_scan_smem(int *input, int size) {
    int tid = threadIdx.x;
    extern __shared__ int shared_mem[];
    if (tid < size) {
        shared_mem[tid] = input[tid];
        __syncthreads();
        for (int stride = 1; stride <= tid; stride *= 2) {
            shared_mem[tid] += shared_mem[tid - stride];
            __syncthreads();
        }
        input[tid] = shared_mem[tid];
    }
}

__global__ void inclusive_scan_post(int *input, int *aux, int size) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < size and blockIdx.x > 0) {
        input[gid] += aux[blockIdx.x - 1];
    }
}

int main(int argc, char **argv) {
    int input_size = 1 << 10;
    if (argc > 1) {
        input_size = 1 << atoi(argv[1]);
    }

    const int byte_size = sizeof(int) * input_size;
    int * h_input, *h_output, *h_ref, *h_aux;

    h_input = (int*)malloc(byte_size);
    h_output = (int*)malloc(byte_size);
    h_ref = (int*)malloc(byte_size);
    
    // randomly initialize the input data
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (int)(rand() & 0xFF);
    }
	inclusive_scan_cpu(h_input, h_output, input_size);

    int *d_input, *d_aux;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int n_blocks = (input_size + block_size - 1) / block_size;
    dim3 block(block_size);
    dim3 grid(n_blocks);
    inclusive_scan_naive <<<grid, block>>>(d_input, input_size);
    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

    printf("inclusive_scan_naive\n");
    compare_arrays(h_output, h_ref, input_size);

    int aux_byte_size = sizeof(int) * n_blocks;
    h_aux = (int*)malloc(aux_byte_size);
    cudaMalloc((void**)&d_aux, sizeof(int) * n_blocks);
    cudaMemcpy(d_aux, h_aux, aux_byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    inclusive_scan_pre<<<grid, block, block_size * sizeof(int)>>>(d_input, d_aux, input_size);
    inclusive_scan_smem<<<1, n_blocks, n_blocks * sizeof(int)>>>(d_aux, n_blocks);
    inclusive_scan_post<<<grid, block>>>(d_input, d_aux, input_size);
    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

    printf("inclusive_scan_efficient\n");
    compare_arrays(h_output, h_ref, input_size);

    // for (int i = 0; i < input_size; i++) {
    //     printf("%d ", h_input[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < input_size; i++) {
    //     printf("%d ", h_ref[i]);
    // }
    // printf("\n");



    free(h_input);
    free(h_output);
    free(h_ref);
    free(h_aux);
    cudaFree(d_input);
    cudaFree(d_aux);

    return 0;
}