#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>

using std::cout;
using std::endl;
using std::vector;

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMultiplication(const int *__restrict a, const int *__restrict b,
                                     int *__restrict c, int n) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * n + col] = 0;
    for (int k = 0; k < n; k++)
    {
        // Accumulate results for a single element
        c[row * n + col] += a[row * n + k] * b[k * n + col];
    }
}

// matrix multiply on the CPU
void matrixMultiplicationCPU(const vector<int> &a, const vector<int> &b, vector<int> &c, int n) {
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            c[row * n + col] = 0;
            for (int i = 0; i < n; ++i)
            {
                c[row * n + col] += a[row * n + i] * b[i * n + col];
            }
        }
    }
}

// verify that the GPU result matches the CPU result
bool verifyResult(const vector<int> &a, const vector<int> &b, int n) {
    for (int i = 0; i < n * n; ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix size of 1024 x 1024;
    int N = 1 << 10;

    // Size (in bytes) of matrix
    size_t bytes = N * N * sizeof(int);

    // Host vectors
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);
    vector<int> ref_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []()
             { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []()
             { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    matrixMultiplication<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    matrixMultiplicationCPU(h_a, h_b, ref_c, N);

    if (verifyResult(h_c, ref_c, N)) {
        cout << "the result is correct" << endl;
    }
    else {
        cout << "the result is incorrect" << endl;
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
