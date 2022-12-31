#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;

__global__ void matrixMultiplication(const int* a, const int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int temp = 0;
        for (int i = 0; i < n; ++i) {
            // 修改
            temp += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = temp;
    }
}

// matrix multiply on the CPU
void matrixMultiplicationCPU(const vector<int>& a, const vector<int>& b, vector<int>& c, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            c[row * n + col] = 0;
            for (int i = 0; i < n; ++i) {
                c[row * n + col] += a[row * n + i] * b[i * n + col];
            }
        }
    }
}

// verify that the GPU result matches the CPU result
bool verifyResult(const vector<int>& a, const vector<int>& b, int n) {
    for (int i = 0; i < n * n; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    // matrix size is 1024 x 1024
    int N = 1 << 10;

    // number of bytes per matrix
    size_t bytes = N * N * sizeof(int);

    // Host memory
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);
    vector<int> ref_c(N * N);

    // initialize matrices on the host to random integers
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Device memory
    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from the host to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // number of threads per thread block
    int THREADS = 32;
    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // execute the kernel
    matrixMultiplication<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // copy the result back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // verify that the GPU result is correct
    matrixMultiplicationCPU(h_a, h_b, ref_c, N);
    if (verifyResult(h_c, ref_c, N)) {
        cout << "The result is correct!";
    } else {
        cout << "The result is incorrect!";
    }
    cout << endl;
    
    // // print some value randomly
    // for (int i = 0; i < 10; ++i) {
    //     int row = rand() % N;
    //     int col = rand() % N;
    //     cout << "h_c[" << row << "][" << col << "] = " << h_c[row * N + col] << endl;
    //     cout << "ref_c[" << row << "][" << col << "] = " << ref_c[row * N + col] << endl;
    // }

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}