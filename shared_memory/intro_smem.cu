#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define SHARED_ARRAY_SIZE 128

__global__ void smem_static_test(int *in, int *out, int size) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_array[SHARED_ARRAY_SIZE];

    if (gid < size) {
        shared_array[tid] = in[gid];
        out[gid] = shared_array[tid];
    }
}

__global__ void smem_dynamic_test(int *in, int *out, int size) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared_array[];

    if (gid < size) {
        shared_array[tid] = in[gid];
        out[gid] = shared_array[tid];
    }
}



int main(int argc, char **argv)
{
    int size = 1 << 22;
    int block_size = SHARED_ARRAY_SIZE;
    bool dynamic = true;
    
    if (argc > 1) {
        dynamic = atoi(argv[1]);
    }

    size_t NO_BYTES = size * sizeof(int);

    int *h_in, *h_ref, *d_in, *d_out;
    h_in = (int *)malloc(NO_BYTES);
    h_ref = (int *)malloc(NO_BYTES);
    // random initialisation of input
    for (int i = 0; i < size; i++) {
        h_in[i] = rand() % 100;
    }

    cudaMalloc((void **)&d_in, NO_BYTES);
    cudaMalloc((void **)&d_out, NO_BYTES);

    dim3 block(block_size);
    dim3 grid(size / block.x + 1);    

    cudaMemcpy(d_in, h_in, NO_BYTES, cudaMemcpyHostToDevice);

    if (dynamic) {
        printf("Dynamic shared memory test");
        smem_dynamic_test << <grid, block, block_size * sizeof(int) >> > (d_in, d_out, size);
    }
    else {
        printf("Static shared memory test");
        smem_static_test << <grid, block >> > (d_in, d_out, size);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_ref, d_out, NO_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_ref);

    cudaDeviceReset();

    return 0;
}