#include <iostream>
#include <cuda_profiler_api.h>

__global__ void increment(int* g_data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

bool is_correct(int* data, int n, int x) {
    for (int i = 0; i < n; i++) {
        if (data[i] != x) {
            printf("Error: data[%d] != %d\n", i, x);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int devId = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&devId);

    cudaGetDeviceProperties(&prop, devId);
    printf("Device: %s\n", prop.name);

    constexpr int n = 16 << 20;
    constexpr int nbytes = n * sizeof(int);
    int value = 26;

    int* a = 0;
    cudaMallocHost(&a, nbytes);
    memset(a, 0, nbytes);

    int* d_a = 0;
    cudaMalloc(&d_a, nbytes);
    cudaMemset(d_a, 255, nbytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);
    cudaDeviceSynchronize();

    float elapsed = 0.0f;
    cudaProfilerStart();
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice);
    increment<<<(n + 255) / 256, 256>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaProfilerStop();

    uint32_t counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Elapsed time: %f ms\n", elapsed);
    printf("Counter: %d\n", counter);

    if (is_correct(a, n, value)) {
        printf("Correct!\n");
    }

    cudaFree(d_a);
    cudaFreeHost(a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}