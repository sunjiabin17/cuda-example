#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>


constexpr int M = 1 << 9;
constexpr int N = 1 << 6;
constexpr int K = 1 << 8;

// constexpr int THREADS = 32;
constexpr int tile_M = 1;  // 32
constexpr int tile_N = 1;  // 32
constexpr int tile_K = 64;

constexpr int BLOCK_M = M / tile_M;
constexpr int BLOCK_N = N / tile_N;


__global__ void gemm(int* A, int* B, int* C) {
    __shared__ int sA[tile_M*tile_K];
    __shared__ int sB[tile_K*tile_N];
    // 每个block负责C中一个维度bm*bn为的小矩阵块的计算
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int temp = 0;
    for (int i = 0; i < K / tile_K; ++i) {
        // 方法1: 计算要copy到shared memory中的A和B的全局地址
        if (tx < tile_K)
        sA[ty * tile_K + tx] = A[(by * tile_M + ty) * K + (i * tile_K + tx)];
        if (ty < tile_K)
        sB[ty * tile_N + tx] = B[(i * tile_K + ty) * N + (bx * tile_N + tx)];
       
        __syncthreads();
        for (int j = 0; j < tile_K; ++j) {
            temp += sA[ty * tile_K + j] * sB[j * tile_N + tx];
        } 
        __syncthreads();
    }

    C[(by * tile_M + ty) * N + (bx * tile_N + tx)] = temp;

    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int c_value = 0;

    // for (int i = 0; i < K; i += blockDim.x) {
    //     sA[threadIdx.y * blockDim.x + threadIdx.x] = A[row * K + i + threadIdx.x];
    //     sB[threadIdx.y * blockDim.x + threadIdx.x] = B[i * N + threadIdx.y * N + col];

    //     __syncthreads();

    //     // 计算当前小块的结果 (blockDim.x = 32)
    //     for (int j = 0; j < blockDim.x; ++j) {
    //         c_value += sA[threadIdx.y * blockDim.x + j] * sB[j * blockDim.x + threadIdx.x];
    //     }
    
    //     __syncthreads();
    // }
    // C[row * N + col] = c_value;

}

int main() {
    int* A = new int[M * K];
    int* B = new int[K * N];
    int* C = new int[M * N];

    std::generate(A, A + M * K, []() {return (int)(rand() % 10); });
    std::generate(B, B + K * N, []() {return (int)(rand() % 10); });
    std::fill(C, C + M * N, 0);

    // std::fill(A, A + M * K, 0.121f);
    // std::fill(B, B + K * N, 1.221f);
    // std::fill(C, C + M * N, 0.0f);

    int* dA, * dB, * dC;
    cudaMalloc(&dA, M * K * sizeof(int));
    cudaMalloc(&dB, K * N * sizeof(int));
    cudaMalloc(&dC, M * N * sizeof(int));

    cudaMemcpy(dA, A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(tile_N, tile_M);
    dim3 grid(BLOCK_N, BLOCK_M);

    gemm<<<grid, block>>>(dA, dB, dC);

    cudaMemcpy(C, dC, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // gemm on cpu
    bool flag = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int c = 0;
            for (int k = 0; k < K; ++k) {
                c += A[i * K + k] * B[k * N + j];
            }
            if (abs(C[i * N + j] - c) > 1) {
                flag = false;
                std::cout << i << " " << j << " " << C[i * N + j] << " " << c << std::endl;
                break;
            }
        }
        if (!flag) {
            break;
        }
    }
    if (flag) {
        std::cout << "correct" << std::endl;
    }
    else {
        std::cout << "wrong" << std::endl;
    }
    // // print matrix
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         int c = 0;
    //         for (int k = 0; k < K; ++k) {
    //             c += A[i * K + k] * B[k * N + j];
    //             std::cout << A[i * K + k] << "*" << B[k * N + j] << "+";
    //         } 
    //         std::cout << "||" << i << " " << j << " " << C[i * N + j] << " " << c << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
