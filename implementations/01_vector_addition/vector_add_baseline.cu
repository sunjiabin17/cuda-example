#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>


__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
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
    constexpr int N = 1 << 16; // 65536 elements
    constexpr size_t bytes = sizeof(int) * N;

    int *h_a, *h_b, *h_c, *ref_c;
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    ref_c = (int*)malloc(bytes);

    init(h_a, N);
    init(h_b, N);

    for (int i = 0; i < N; i++) {
        ref_c[i] = h_a[i] + h_b[i];
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 1 << 10;
    
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    // int NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    verify(h_c, ref_c, N);

    free(h_a);
    free(h_b);
    free(h_c);
    free(ref_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// compile: nvcc -o baseline thisfile.cu
// run: ./baseline
// profile: nvprof ./baseline
