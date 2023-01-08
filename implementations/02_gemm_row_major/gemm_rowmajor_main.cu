#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "gemm1_naive.cuh"
#include "gemm2_tile.cuh"
#include "gemm3_4x1.cuh"

using namespace std;

#define M 1024
#define N 2048
#define K 4096

int main(int argc, char **argv) {

    int prints = 0;
    if (argc > 1) {
        prints = atoi(argv[1]);
    }


    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C1 = new float[M * N];

    // set random seed
    // srand((unsigned)time(NULL));

    std::generate(A, A + M * K, []() {return (float)(rand() % 10); });
    std::generate(B, B + K * N, []() {return (float)(rand() % 10); });
    std::fill(C, C + M * N, 0.0f);

    float* dA, * dB, * dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = K;
    int ldb = N;
    int ldc = N;

    // matrix multiplication using cublas (M, K) * (K, N) = (M, N)
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, ldb, dA, lda, &beta, dC, ldc);
    cudaMemcpy(C1, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    test_sgemm1(M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    test_sgemm2(M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    test_sgemm3(M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemset(dC, 0.f, M * N * sizeof(float));


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

    if (prints > 0) {
        // // print A
        // std::cout << "A: " << std::endl;
        // for (int i = 0; i < M; ++i) {
        //     for (int j = 0; j < K; ++j) {
        //         std::cout << A[i * K + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        // // print B
        // std::cout << "B: " << std::endl;
        // for (int i = 0; i < K; ++i) {
        //     for (int j = 0; j < N; ++j) {
        //         std::cout << B[i * N + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        // print C
        int cnt = 0;
        std::cout << "C: " << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (cnt > prints)
                    break;
                std::cout << C[i * N + j] << " ";
                cnt++;
            }
            if (cnt > prints)
                break;
            std::cout << std::endl;
        }

        cnt = 0;
        std::cout << std::endl;
        // print C1
        std::cout << "C1: " << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (cnt > prints)
                    break;
                std::cout << C1[i * N + j] << " ";
                cnt++;
            }
            if (cnt > prints)
                break;
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