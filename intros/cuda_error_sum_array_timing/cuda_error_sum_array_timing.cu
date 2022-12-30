#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void CHECK_ERROR(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("Error: %s, %s, %d", cudaGetErrorString(err), __FILE__, __LINE__);
    }
}

__global__ void add_on_gpu(int *a, int *b, int *c, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        c[tid] = a[tid] + b[tid];
}

void add_on_cpu(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
        c[i] = a[i] + b[i];
}

void check(int *c, int* c2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (c[i] != c2[i])
        {
            printf("Error at %d: %d != %d", i, c[i], c2[i]);
            return;
        }
    }
    printf("Success!");
}

int main() {
    int *a, *b, *cpu_result, *gpu_result;
    int *dev_a, *dev_b, *dev_c;
    int size = (1 << 25);
    int blocksize = 256;
    int byte_size = size * sizeof(int);
    a = (int*)malloc(byte_size);
    b = (int*)malloc(byte_size);
    cpu_result = (int*)malloc(byte_size);
    gpu_result = (int*)malloc(byte_size);
    memset(cpu_result, 0, byte_size);
    memset(gpu_result, 0, byte_size);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        a[i] = (int)(rand() & 0xff);
        b[i] = (int)(rand() & 0xff);
    }
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    add_on_cpu(a, b, cpu_result, size);
    cpu_end = clock();
    printf("CPU time: %f \n", (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);

    CHECK_ERROR(cudaMalloc((void**)&dev_a, byte_size));
    CHECK_ERROR(cudaMalloc((void**)&dev_b, byte_size));
    CHECK_ERROR(cudaMalloc((void**)&dev_c, byte_size));
    // cudaMalloc((void**)&dev_a, byte_size);
    // cudaMalloc((void**)&dev_b, byte_size);
    // cudaMalloc((void**)&dev_c, byte_size);

    clock_t gpu_h2d_start, gpu_h2d_end;
    gpu_h2d_start = clock();
    CHECK_ERROR(cudaMemcpy(dev_a, a, byte_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_b, b, byte_size, cudaMemcpyHostToDevice));
    // cudaMemcpy(dev_a, a, byte_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_b, b, byte_size, cudaMemcpyHostToDevice);
    gpu_h2d_end = clock();
    printf("GPU H2D time: %f \n", (double)(gpu_h2d_end - gpu_h2d_start) / CLOCKS_PER_SEC);
    
    dim3 block(blocksize);
    dim3 grid((size / blocksize) + 1);

    clock_t gpu_kernel_start, gpu_kernel_end;
    gpu_kernel_start = clock();
    add_on_gpu << <grid, block >> > (dev_a, dev_b, dev_c, size);
    
    CHECK_ERROR(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();
    gpu_kernel_end = clock();
    printf("GPU kernel time: %f \n", (double)(gpu_kernel_end - gpu_kernel_start) / CLOCKS_PER_SEC);

    clock_t gpu_d2h_start, gpu_d2h_end;
    gpu_d2h_start = clock();
    CHECK_ERROR(cudaMemcpy(gpu_result, dev_c, byte_size, cudaMemcpyDeviceToHost));
    // cudaMemcpy(gpu_result, dev_c, byte_size, cudaMemcpyDeviceToHost);
    gpu_d2h_end = clock();
    printf("GPU D2H time: %f \n", (double)(gpu_d2h_end - gpu_d2h_start) / CLOCKS_PER_SEC);
    
    check(cpu_result, gpu_result, size);

    CHECK_ERROR(cudaFree(dev_a));
    CHECK_ERROR(cudaFree(dev_b));
    CHECK_ERROR(cudaFree(dev_c));
    // cudaFree(dev_a);
    // cudaFree(dev_b);
    // cudaFree(dev_c);

    free(a);
    free(b);
    free(cpu_result);
    free(gpu_result);

    CHECK_ERROR(cudaDeviceReset());
    // cudaDeviceReset();

    return 0;
}