#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include<stdio.h>
#include<stdlib.h>

#include "ref_gemm.cuh"
#include "gemm1_naive.cuh"
#include "gemm2_tile.cuh"
#include "gemm3_tile.cuh"
#include "gemm4_reduce_bank_conflict.cuh"
#include "gemm5_reduce_bank_conflict.cuh"
#include "gemm6_4x1_micro_kernel.cuh"
#include "gemm7_4x1_vectorized_load_store.cuh"
#include "gemm8_4x4_micro_kernel.cuh"
#include "gemm9_8x8_micro_kernel.cuh"
#include "gemm10_8x8_warp_tile.cuh"
#include "gemm11_8x8_prefetch.cuh"
#include "gemm12_8x8_double_buffer.cuh"


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./a.out M N K" << std::endl;
        return 0;
    }

    bool is_print = false;
    if (argc == 5) {
        is_print = true;
    }

    int M = atoi(argv[1]);
    M = (int)pow(2, M);
    int N = atoi(argv[2]);
    N = (int)pow(2, N);
    int K = atoi(argv[3]);
    K = (int)pow(2, K);

    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

    // set random seed
    srand((unsigned)time(NULL));

    std::generate(A, A + M * K, []() {return (float)(rand() % 10); });
    std::generate(B, B + K * N, []() {return (float)(rand() % 10); });
    std::fill(C, C + M * N, 0.0f);

    // std::fill(A, A + M * K, 0.121f);
    // std::fill(B, B + K * N, 1.221f);
    // std::fill(C, C + M * N, 0.0f);

    float* dA, * dB, * dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_M = (M + bM - 1) / bM;
    int BLOCK_N = (N + bN - 1) / bN;
    int BLOCK_K = (K + bK - 1) / bK;

    dim3 block(32, 32);
    dim3 grid(BLOCK_M, BLOCK_N);
    dim3 grid2((M+64-1)/64, (N+64-1)/64);
    dim3 block9(16, 16);
    dim3 grid3((M+128-1)/128, (N+128-1)/128);

    gemm1_naive<<<grid, block>>>(M, N, K, dA, dB, dC);
    gemm2_tile<<<grid, block>>>(M, N, K, dA, dB, dC);
    gemm3_tile<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm4_reduce_bank_conflict<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm5_reduce_bank_conflict<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm6_4x1_micro_kernel<<<grid, 16*16>>>(M, N, K, dA, dB, dC);   // 每个线程处理4个元素
    gemm7_4x1_vectorized_load_store<<<grid, 16*16>>>(M, N, K, dA, dB, dC);   // 每个线程处理4个元素
    gemm8_4x4_micro_kernel<<<grid2, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4x4个元素
    gemm8_4x4_micro_kernel_2<<<grid2, block9>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4x4个元素
    gemm9_8x8_micro_kernel<<<grid3, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理8x8个元素
    gemm10_8x8_warp_tile<<<grid3, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理8x8个元素, warp tile
    gemm11_8x8_prefetch<<<grid3, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理8x8个元素, warp tile, prefetch
    gemm12_8x8_double_buffer<<<grid3, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理8x8个元素, warp tile, double buffer
    // mysgemm_v1<<<grid, block>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    // mysgemm_v2<<<grid, block>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    // mysgemm_v3<<<grid, 32*32>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    // mysgemm_v4<<<grid, 32*32>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    // mysgemm_v5<<<grid, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4个元素
    // mysgemm_v7<<<grid2, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4x4个元素
    // mysgemm_v8<<<grid3, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理8x8个元素
    // mysgemm_v9<<<grid3, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理8x8个元素
    cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);    
    
    // matrix multiplication using cublas (M, K) * (K, N) = (M, N)
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);

    // // 方法1
    // // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, dA, K, dB, N, &beta, dC, M);
    // // 方法2
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    
    float* C1 = new float[M * N];
    cudaMemcpy(C1, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // compare
    bool flag = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(C[i * N + j] - C1[i * N + j]) > 1e-5) {
                std::cout << i << " " << j << " " << C[i * N + j] << " " << C1[i * N + j] << std::endl;
                flag = false;
                break;
            }
        }
        if (!flag) {
            break;
        }
    }
    if (flag) {
        std::cout << "correct" << std::endl;
    } else {
        std::cout << "wrong" << std::endl;
    }

    if (is_print) {
        // print A
        std::cout << "A: " << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                std::cout << A[i * K + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        // print B
        std::cout << "B: " << std::endl;
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << B[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        // print C
        std::cout << "C: " << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        // print C1
        std::cout << "C1: " << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << C1[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] A;
    delete[] B;
    delete[] C;



    return 0;
}
