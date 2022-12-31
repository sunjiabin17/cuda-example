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
    constexpr int N = 1 << 16; // 65536 elements
    constexpr size_t bytes = sizeof(int) * N;

    int *h_a, *h_b, *h_c, *ref_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    
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
    
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(ref_c);
    
    return 0;
}

// compile: nvcc -o pinned thisfile.cu
// run: ./pinned
// profile: nvprof ./pinned
// nvprof --print-gpu-trace ./pinned
/***
(base) ➜  vector_addition git:(main) ✗ nvprof --print-gpu-trace ./pinned
==9319== NVPROF is profiling process 9319, command: ./pinned
Success!==9319== Profiling application: ./pinned
==9319== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
212.39ms  23.298us                    -               -         -         -         -  256.00KB  10.479GB/s      Pinned      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
212.42ms  22.914us                    -               -         -         -         -  256.00KB  10.655GB/s      Pinned      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
212.45ms  23.650us                    -               -         -         -         -  256.00KB  10.323GB/s      Pinned      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
212.49ms  7.8410us             (64 1 1)      (1024 1 1)         8        0B        0B         -           -           -           -  NVIDIA GeForce          1         7  vectorAdd(int*, int*, int*, int) [120]
212.50ms  20.738us                    -               -         -         -         -  256.00KB  11.773GB/s      Device      Pinned  NVIDIA GeForce          1         7  [CUDA memcpy DtoH]
 * ***/