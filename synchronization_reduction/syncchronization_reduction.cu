#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void scan_efficient_1G(int *input, int *auxiliry_array, int input_size);
__global__ void scan_summation(int *input, int *auxiliry_array, int input_size);

// reduction neighbored pairs kernel
__global__ void redunction_neighbored_pairs(int *input,
                                            int *temp, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid > size)
        return;

    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
    {
        if (tid % (2 * offset) == 0)
        {
            input[gid] += input[gid + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp[blockIdx.x] = input[gid];
    }
}

// warp divergence improved
__global__ void reduction_neighbored_pairs_improved(
    int *int_array, int *temp_array, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // local data block pointer
    int *i_data = int_array + blockDim.x * blockIdx.x;

    if (gid > size)
        return;

    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
    {
        int index = 2 * offset * tid;

        if (index < blockDim.x)
        {
            i_data[index] += i_data[index + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp_array[blockIdx.x] = int_array[gid];
    }
}

//kernel interleaved pairs
__global__ void reduction_interleaved_pairs(int * int_array,
    int * temp_array, int size)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid > size)
        return;

    for (int offset = blockDim.x/ 2; offset > 0; offset = offset/2)
    {
        if (tid < offset)
        {
            int_array[gid] += int_array[gid + offset];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        temp_array[blockIdx.x] = int_array[gid];
    }
}

// loop unrolling
__global__ void reduction_kernel_warp_unrolling(int * int_array,
    int * temp_array, int size)
{
    int tid = threadIdx.x;

    //element index for this thread
    int index = blockDim.x * blockIdx.x  + threadIdx.x;

    //local data pointer
    int * i_data = int_array + blockDim.x * blockIdx.x ;

    for (int offset = blockDim.x/2; offset >= 64; offset = offset/2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }
		// 展开 unrolling last 5 iteration
    if (tid < 32)
    {
        volatile int * vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        temp_array[blockIdx.x] = i_data[0];
    }
}

// warp unrolling
__global__ void reduction_kernel_warp_unrolling(int * int_array,
    int * temp_array, int size)
{
    int tid = threadIdx.x;

    //element index for this thread
    int index = blockDim.x * blockIdx.x  + threadIdx.x;

    //local data pointer
    int * i_data = int_array + blockDim.x * blockIdx.x ;

    for (int offset = blockDim.x/2; offset >= 64; offset = offset/2)
    {
        if (tid < offset)
        {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }
		// 展开 unrolling last 5 iteration
    if (tid < 32)
    {
        volatile int * vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0)
    {
        temp_array[blockIdx.x] = i_data[0];
    }
}

int main(int argc, char **argv)
{
    printf("Running neighbored pairs reduction kernel \n");

    int size = 1 << 27; // 128 Mb of data
    int byte_size = size * sizeof(int);
    int block_size = 128;

    int *h_input, *h_ref;
    h_input = (int *)malloc(byte_size);

    initialize(h_input, size, INIT_RANDOM);

    // get the reduction result from cpu
    int cpu_result = reduction_cpu(h_input, size);

    dim3 block(block_size);
    dim3 grid(size / block.x);

    printf("Kernel launch parameters | grid.x : %d, block.x : %d \n",
           grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x;
    h_ref = (int *)malloc(temp_array_byte_size);

    int *d_input, *d_temp;

    gpuErrchk(cudaMalloc((void **)&d_input, byte_size));
    gpuErrchk(cudaMalloc((void **)&d_temp, temp_array_byte_size));

    gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
    gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
                         cudaMemcpyHostToDevice));

    redunction_neighbored_pairs<<<grid, block>>>(d_input, d_temp, size);

    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(h_ref, d_temp, temp_array_byte_size,
               cudaMemcpyDeviceToHost);

    int gpu_result = 0;

    for (int i = 0; i < grid.x; i++)
    {
        gpu_result += h_ref[i];
    }

    // validity check
    compare_results(gpu_result, cpu_result);

    gpuErrchk(cudaFree(d_temp));
    gpuErrchk(cudaFree(d_input));

    free(h_ref);
    free(h_input);

    gpuErrchk(cudaDeviceReset());
    return 0;
}