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

void exclusive_scan_cpu(int *input, int *output, int size) {
    output[0] = 0;
    for (int i = 1; i < size; i++)
    {
        output[i] = output[i - 1] + input[i - 1];
    }
}

__global__ void efficient_exclusive_scan_single_block(int *input, int size) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        // reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = (tid + 1) * stride * 2 - 1;	// odd index
            if (index < blockDim.x) {
                input[index] += input[index - stride];
            }
            __syncthreads();
        }

        // set root value to 0
        if (tid == 0) {
            input[blockDim.x - 1] = 0;
        }

        int temp = 0;

        // down-sweep phase
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            int index = (tid + 1) * stride * 2 - 1;
            if (index < blockDim.x) {
                temp = input[index - stride];	// assign left child value to temp
                input[index - stride] = input[index];	// assign parent value to left child
                input[index] += temp;	// add left child value to parent value
            }
            __syncthreads();
        }
    }
}

__global__ void efficient_exclusive_scan_single_block_smem(int *input, int size) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        extern __shared__ int shared_mem[];
        shared_mem[tid] = input[gid];
        __syncthreads();
        // reduction phase
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int idx = (tid + 1) * stride * 2 - 1;	// odd index
            if (idx < blockDim.x) {
                shared_mem[idx] += shared_mem[idx - stride];
            }
            __syncthreads();
        }

        // set root value to 0
        if (tid == 0) {
            shared_mem[blockDim.x - 1] = 0;
        }

        int temp = 0;
        // down-sweep phase
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            int idx = (tid + 1) * stride * 2 - 1;
            if (idx < blockDim.x) {
                temp = shared_mem[idx - stride];	// assign left child value to temp
                shared_mem[idx - stride] = shared_mem[idx];	// assign parent value to left child
                shared_mem[idx] += temp;	// add left child value to parent value
            }
            __syncthreads();
        }
        // write back to global memory
        input[gid] = shared_mem[tid];
    }
}

// __global__ void efficient_exclusive_scan_pre(int *input, int* aux, int size) {
//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (gid < size)
//     {
//         extern __shared__ int shared_mem[];
//         shared_mem[tid] = input[gid];
//         __syncthreads();
//         // reduction phase
//         for (int stride = 1; stride < blockDim.x; stride *= 2) {
//             int idx = (tid + 1) * stride * 2 - 1;	// odd index
//             if (idx < blockDim.x) {
//                 shared_mem[idx] += shared_mem[idx - stride];
//             }
//             __syncthreads();
//         }

//         // set root value to 0
//         if (tid == 0) {
//             shared_mem[blockDim.x - 1] = 0;
//         }

//         int temp = 0;
//         // down-sweep phase
//         for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//             int idx = (tid + 1) * stride * 2 - 1;
//             if (idx < blockDim.x) {
//                 temp = shared_mem[idx - stride];	// assign left child value to temp
//                 shared_mem[idx - stride] = shared_mem[idx];	// assign parent value to left child
//                 shared_mem[idx] += temp;	// add left child value to parent value
//             }
//             __syncthreads();
//         }
//         // write back to global memory
//         input[gid] = shared_mem[tid];
//         if (tid == 0) {
//             aux[blockIdx.x] = shared_mem[blockDim.x - 1];
//         }
//     }
// }

// __global__ void efficient_exclusive_scan_post(int *input, int* aux, int size) {
//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (gid < size and blockIdx.x > 0)
//     {
//         input[gid] += aux[blockIdx.x - 1];
//     }
// }

// __global__ void inclusive_scan_single_block(int *input, int size) {
//     int tid = threadIdx.x;
//     extern __shared__ int shared_mem[];
//     if (tid < size) {
//         shared_mem[tid] = input[tid];
//         __syncthreads();
//         for (int stride = 1; stride <= tid; stride *= 2) {
//             shared_mem[tid] += shared_mem[tid - stride];
//             __syncthreads();
//         }
//         input[tid] = shared_mem[tid];
//     }
// }

int main(int argc, char **argv) {
    int input_size = 1 << 4;
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
	exclusive_scan_cpu(h_input, h_output, input_size);

    int *d_input, *d_aux;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    int block_size = input_size / 8;
    int n_blocks = (input_size + block_size - 1) / block_size;
    dim3 block(block_size);
    dim3 grid(n_blocks);
    // efficient_exclusive_scan_single_block<<<1, input_size>>>(d_input, input_size);
    efficient_exclusive_scan_single_block_smem<<<1, input_size, block_size * sizeof(int)>>>(d_input, input_size);
    cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

    printf("exclusive_scan_naive\n");
    compare_arrays(h_output, h_ref, input_size);

    // int aux_byte_size = sizeof(int) * n_blocks;
    // h_aux = (int*)malloc(aux_byte_size);
    // cudaMalloc((void**)&d_aux, sizeof(int) * n_blocks);
    // cudaMemcpy(d_aux, h_aux, aux_byte_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // // execute the kernel
    // efficient_exclusive_scan_pre<<<grid, block, block_size * sizeof(int)>>>(d_input, d_aux, input_size);
    // inclusive_scan_single_block<<<1, n_blocks, n_blocks * sizeof(int)>>>(d_aux, n_blocks);
    // efficient_exclusive_scan_post<<<grid, block>>>(d_input, d_aux, input_size);
    // cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

    // printf("exclusive_scan_efficient\n");
    // compare_arrays(h_output, h_ref, input_size);


    // for (int i = 0; i < input_size; i++) {
    //     printf("%d ", h_input[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < input_size; i++) {
    //     printf("%d ", h_output[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < input_size; i++) {
    //     printf("%d ", h_ref[i]);
    // }
    // printf("\n");

    free(h_input);
    free(h_output);
    free(h_ref);
    // free(h_aux);
    cudaFree(d_input);
    cudaFree(d_aux);

    return 0;
}