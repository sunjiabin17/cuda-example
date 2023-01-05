#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include<stdio.h>
#include<stdlib.h>


#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa(i,j) sa[((i)<<5) + (j)]
#define sb(i,j) sb[((i)<<5) + (j)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
__global__  __launch_bounds__(1024)
void mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa(tx,ty)=A(tx,ty);
        sb(ty,tx)=B(tx,ty);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            tmp += sa(tx,inner_k_count) * sb(ty,inner_k_count);
        }
        __syncthreads();
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa4(i,j) sa4[((j)<<5) + (i)]
#define sb4(i,j) sb4[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
__global__  __launch_bounds__(1024)
void mysgemm_v4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa4[MS*KS];
    __shared__ float sb4[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa4(row,col)=A(row,col);
        sb4(col,row)=B(row,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            tmp += sa4(row,inner_k_count) * sb4(col,inner_k_count);
        }
        __syncthreads();
    }
    C(row,col) = alpha * tmp + beta*C(row,col);
}

__global__ __launch_bounds__(1024) void gemm1_naive(int M, int N, int K, float* A, float* B, float* C) {
    // 列优先
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    float tmp = 0.0f;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);
    for (int k_idx = 0; k_idx < K; k_idx++) {
        tmp += A[tx + k_idx * lda] * B[k_idx + ty * ldb];
    }
    C[tx + ty * ldc] = tmp;
}

#define bM 32
#define bN 32
#define bK 32
__global__ __launch_bounds__(1024) void gemm2_tile(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    // A = &(A[(bx << 5)]);
    // B = &(B[(by << 5) * ldb]);
    // C = &(C[(bx << 5) + (by << 5) * ldc]);
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    
    // update: tile share memory
    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmp = 0.0f;
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[tx * bK + ty] = A[tx + ty * lda]; // bank conflict
        b_tile[ty * bK + tx] = B[tx + ty * ldb];
        A += (lda << 5);
        B += bK;
        __syncthreads();
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            tmp += a_tile[tx * bK + inner_k] * b_tile[ty * bK + inner_k];
        }
        __syncthreads(); 
    }
    C[tx + ty * ldc] = 1.f * tmp + 0.f * C[tx + ty * ldc];

}

__global__ __launch_bounds__(1024) void gemm3_tile(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
     // update: one dim block
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmp = 0.0f;
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[col + row * bK] = A[row + col * lda]; // bank conflict
        b_tile[row + col * bK] = B[row + col * ldb];
        A += (lda << 5);
        B += bK;
        __syncthreads();
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            tmp += a_tile[inner_k + row * bK] * b_tile[inner_k + col * bK];
        }
        __syncthreads(); 
    }
    C[row + col * ldc] = tmp;
}

__global__ __launch_bounds__(1024) void gemm4_reduce_bank_conflict(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim block
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmp = 0.0f;
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[row + col * bK] = A[row + col * lda];    // update
        b_tile[col + row * bK] = B[row + col * ldb];
        A += (lda << 5);
        B += bK;
        __syncthreads();
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            // memory coelascing on shared memory
            tmp += a_tile[row + inner_k * bK] * b_tile[col + inner_k * bK]; // update
        }
        __syncthreads();
    }
    C[row + col * ldc] = tmp;
}

__global__ __launch_bounds__(1024) void gemm5_reduce_bank_conflict(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim block
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmp = 0.0f;
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[row + col * bK] = A[row + col * lda];    // update
        b_tile[row + col * bK] = B[row + col * ldb];    // update b_tile按row连续
        A += (lda << 5);
        B += bK;
        __syncthreads();
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            // memory coelascing on shared memory
            tmp += a_tile[row + inner_k * bK] * b_tile[inner_k + col * bK]; // update
        }
        __syncthreads();
    }
    C[row + col * ldc] = tmp;
}


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

    int BLOCK_M = (M + bM - 1) / bM;
    int BLOCK_N = (N + bN - 1) / bN;
    int BLOCK_K = (K + bK - 1) / bK;


    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

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

    dim3 block(32, 32);
    dim3 grid(BLOCK_M, BLOCK_N);

    gemm1_naive<<<grid, block>>>(M, N, K, dA, dB, dC);
    gemm2_tile<<<grid, block>>>(M, N, K, dA, dB, dC);
    gemm3_tile<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm4_reduce_bank_conflict<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm5_reduce_bank_conflict<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    mysgemm_v2<<<grid, block>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    mysgemm_v4<<<grid, 32*32>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
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
