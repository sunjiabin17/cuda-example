#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

void init(int *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 100;
}

void verify(int *c, int* ref_c, int n) {
    for (int i = 0; i < n; i++) {
        if (c[i] != ref_c[i]) {
            printf("Error at index %d: %d != %d", i, c[i], ref_c[i]);
            return;
        }
    }
    printf("Success!");
}


int main() {
    int device = -1;
    cudaGetDevice(&device);

    constexpr int N = 1 << 16; // 65536 elements
    constexpr size_t bytes = sizeof(int) * N;

    int *a, *b, *c, *ref_c;
    // unified memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    
    ref_c = (int*)malloc(bytes);

    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, device);
    
    init(a, N);
    init(b, N);

    for (int i = 0; i < N; i++) {
        ref_c[i] = a[i] + b[i];
    }

    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, device);
    cudaMemPrefetchAsync(a, bytes, device);
    cudaMemPrefetchAsync(b, bytes, device);

    int NUM_THREADS = 1 << 10;
    
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    // int NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    verify(c, ref_c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(ref_c);
    
    return 0;
}

// compile: nvcc -o prefetch thisfile.cu
// run: ./prefetch
// profile: nvprof ./prefetch
