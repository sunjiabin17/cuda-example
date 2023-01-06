#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include<stdio.h>
#include<stdlib.h>

#include "ref_gemm.cuh"

#define bM 32
#define bN 32
#define bK 32

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

__global__ __launch_bounds__(256) void gemm6_4x1_micro_kernel(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim block 16*16
    int bx = blockIdx.x, by = blockIdx.y;
    // 4个thread处理同一行(一列)数据
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3;
    int col = tx>>3;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmps[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b00;
    
    #define IDX2C(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
    #define SIDX(i,j) IDX2C(i,j,bK)         // shared memory index
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[SIDX(row1, col)] = A[IDX2C(row1, col, lda)];
        a_tile[SIDX(row2, col)] = A[IDX2C(row2, col, lda)];
        a_tile[SIDX(row3, col)] = A[IDX2C(row3, col, lda)];
        a_tile[SIDX(row4, col)] = A[IDX2C(row4, col, lda)];
        // b_tile[SIDX(col, row1)] = B[IDX2C(row1, col, ldb)];
        // b_tile[SIDX(col, row2)] = B[IDX2C(row2, col, ldb)];
        // b_tile[SIDX(col, row3)] = B[IDX2C(row3, col, ldb)];
        // b_tile[SIDX(col, row4)] = B[IDX2C(row4, col, ldb)];
        b_tile[SIDX(row1, col)] = B[IDX2C(row1, col, ldb)];  // 耗时更少
        b_tile[SIDX(row2, col)] = B[IDX2C(row2, col, ldb)];
        b_tile[SIDX(row3, col)] = B[IDX2C(row3, col, ldb)];
        b_tile[SIDX(row4, col)] = B[IDX2C(row4, col, ldb)];
        A += (lda << 5);
        B += bK;
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            // b00 = b_tile[SIDX(col, inner_k)];
            b00 = b_tile[SIDX(inner_k, col)];    // 耗时更少
            tmps[0] += a_tile[SIDX(row1, inner_k)] * b00;
            tmps[1] += a_tile[SIDX(row2, inner_k)] * b00;
            tmps[2] += a_tile[SIDX(row3, inner_k)] * b00;
            tmps[3] += a_tile[SIDX(row4, inner_k)] * b00;
        }
        __syncthreads();
    }

    C[IDX2C(row1, col, ldc)] = tmps[0];
    C[IDX2C(row2, col, ldc)] = tmps[1];
    C[IDX2C(row3, col, ldc)] = tmps[2];
    C[IDX2C(row4, col, ldc)] = tmps[3];
}

__global__ __launch_bounds__(256) void gemm7_4x1_vectorized_load_store(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim block 16*16
    int bx = blockIdx.x, by = blockIdx.y;
    // 4个thread处理同一行(一列)数据
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3;
    int col = tx>>3;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float4 Av, Bv, Cv, tmps;
    tmps.x = 0.0f, tmps.y = 0.0f, tmps.z = 0.0f, tmps.w = 0.0f;
    
    float b00;
    
    #define IDX2C(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
    #define SIDX(i,j) IDX2C(i,j,bK)         // shared memory index
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        Av = *((float4*)&(A[IDX2C(row1, col, lda)]));
        Bv = *((float4*)&(B[IDX2C(row1, col, ldb)]));
        ((float4*)a_tile)[tx] = Av;
        ((float4*)b_tile)[tx] = Bv;     // 耗时更少
        // b_tile[SIDX(col, row1)] = Bv.x;
        // b_tile[SIDX(col, row2)] = Bv.y;
        // b_tile[SIDX(col, row3)] = Bv.z;
        // b_tile[SIDX(col, row4)] = Bv.w;
        A += (lda << 5);
        B += bK;
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK; ++inner_k) {
            // b00 = b_tile[SIDX(col, inner_k)];
            b00 = b_tile[SIDX(inner_k, col)];   // 耗时更少
            tmps.x += a_tile[SIDX(row1, inner_k)] * b00;
            tmps.y += a_tile[SIDX(row2, inner_k)] * b00;
            tmps.z += a_tile[SIDX(row3, inner_k)] * b00;
            tmps.w += a_tile[SIDX(row4, inner_k)] * b00;
        }
        __syncthreads();
    }
    Cv = *((float4*)(&C[IDX2C(row1, col, ldc)]));   // C在(row1, col)位置的地址强转成float4*类型，然后取值给Cv
    Cv.x = tmps.x;
    Cv.y = tmps.y;
    Cv.z = tmps.z;
    Cv.w = tmps.w;
    *((float4*)(&C[IDX2C(row1, col, ldc)])) = Cv; 
}

#define bM8 64
#define bN8 64
#define bK8 16

__global__ __launch_bounds__(256) void gemm8_4x4_micro_kernel(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 64*64, thread size: 16*16
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim
    int bx = blockIdx.x, by = blockIdx.y;
    
    #define IDX8(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
    #define SIDX8(i,j) IDX8(i,j,(1<<6))     // shared memory index

    int row_a = (tx&15)<<2, col_a = tx>>4;  // row_a = range(0, 64, 4), col_a = range(0, 16, 1)
    int row_b = (tx&3)<<2, col_b = tx>>2;   // row_b = range(0, 16, 4), col_b = range(0, 64, 1)
    int col_c = col_a<<2;                   // col_c = range(0, 64, 4)
    int lda16 = lda<<4;     // block size 64, lda * 64

    A = &(A[(bx << 6)]);
    B = &(B[(by << 6) * ldb]);
    C = &(C[(bx << 6) + (by << 6) * ldc]);  // the TB size is 64.

    __shared__ float a_tile[bM8*bK8];
    __shared__ float b_tile[bK8*bN8];

    float4 Av, Bv, Cv[4], tmps[4];
    memset(tmps, 0, sizeof(tmps));

    // 1. 数据布局如何变化: from global memory to shared memory
    // 2. float4加载连续内存如何处理
    // 3. 数据在shared memory中的位置以及如何计算
    for (int k_idx = 0; k_idx < K; k_idx += bK8) {
        Av = *((float4 *)(&A[IDX8(row_a, col_a, lda)]));   // 从A中取连续的4个数
        Bv = *((float4 *)(&B[IDX8(row_b, col_b, ldb)]));   // 从B中取连续的4个数
        ((float4 *)a_tile)[tx] = Av;            // 将Av中的4个数存入共享内存中的连续内存
        // ((float4 *)b_tile)[tx] = Bv;
        b_tile[SIDX8(col_b, row_b)] = Bv.x;     // 将Bv中的4个数存入共享内存, 内存不连续, 之间的距离为64
        b_tile[SIDX8(col_b, row_b+1)] = Bv.y;
        b_tile[SIDX8(col_b, row_b+2)] = Bv.z;
        b_tile[SIDX8(col_b, row_b+3)] = Bv.w;
        A += lda16; // A偏移64列 64*lda (列优先)
        B += bK8;   // B偏移16行
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK8; ++inner_k) {   // bK8=16, 数据宽度为64, 每次计算4*4个数
            Av = *((float4 *)(&a_tile[SIDX8(row_a, inner_k)])); // 从共享内存中取出连续的4个值给Av，需要和b相乘并累加
            Bv = *((float4 *)(&b_tile[SIDX8(col_c, inner_k)])); // 从共享内存中取出连续的4个值给Bv，这4个数对应bK8中每个维度的值
            // 第一次迭代: inner_k=0, 
            // 当线程是0-3时, 
            // Av从shared memory中取的地址为0,1,2,3;4,...,15; (第一列0:32)  (列优先)
            // Bv从shared memory中取的地址为0,1,2,3.0,1,2,3,...,0,1,2,3
            // 当线程是4-7时,
            // Av从shared memory中取的地址为16,17,18,19;20,...,31;(第一列16:32)
            // Bv从shared memory中取的地址为0,1,2,3.0,1,2,3,...,0,1,2,3
            // ...
            // 第二次迭代: inner_k=1, 
            // 当线程是0-3时,
            // Av从shared memory中取的地址为64,65,66,67;68,...,79; (第二列0:32)
            // Bv从shared memory中取的地址为4,5,6,7.4,5,6,7,...,4,5,6,7
            // 当线程是4-7时,
            // Av从shared memory中取的地址为80,81,82,83;84,...,95;(第二列16:32)
            // Bv从shared memory中取的地址为4,5,6,7.4,5,6,7,...,4,5,6,7
            // ...
            tmps[0].x += Av.x * Bv.x;   // tmps[x][y] 对应计算A[x][:] * B[:][y], 结果存在C[x][y]处
            tmps[0].y += Av.y * Bv.x;
            tmps[0].z += Av.z * Bv.x;
            tmps[0].w += Av.w * Bv.x;
            tmps[1].x += Av.x * Bv.y;
            tmps[1].y += Av.y * Bv.y;
            tmps[1].z += Av.z * Bv.y;
            tmps[1].w += Av.w * Bv.y;
            tmps[2].x += Av.x * Bv.z;
            tmps[2].y += Av.y * Bv.z;
            tmps[2].z += Av.z * Bv.z;
            tmps[2].w += Av.w * Bv.z;
            tmps[3].x += Av.x * Bv.w;
            tmps[3].y += Av.y * Bv.w;
            tmps[3].z += Av.z * Bv.w;
            tmps[3].w += Av.w * Bv.w;
        }
        __syncthreads();
    }
    // 从全局内存中取出row_a, col_c处连续的4个值给Cv
    // row_a = range(0, 64, 4), col_c = range(0, 64, 4)
    Cv[0] = *((float4*)(&C[IDX8(row_a, col_c, ldc)]));      // 对应(row_a, col_c), (row_a+1, col_c), (row_a+2, col_c), (row_a+3, col_c)
    Cv[1] = *((float4*)(&C[IDX8(row_a, col_c+1, ldc)]));    
    Cv[2] = *((float4*)(&C[IDX8(row_a, col_c+2, ldc)]));
    Cv[3] = *((float4*)(&C[IDX8(row_a, col_c+3, ldc)]));

    tmps[0].x = alpha * tmps[0].x + beta * Cv[0].x;
    tmps[0].y = alpha * tmps[0].y + beta * Cv[0].y;
    tmps[0].z = alpha * tmps[0].z + beta * Cv[0].z;
    tmps[0].w = alpha * tmps[0].w + beta * Cv[0].w;
    tmps[1].x = alpha * tmps[1].x + beta * Cv[1].x;
    tmps[1].y = alpha * tmps[1].y + beta * Cv[1].y;
    tmps[1].z = alpha * tmps[1].z + beta * Cv[1].z;
    tmps[1].w = alpha * tmps[1].w + beta * Cv[1].w;
    tmps[2].x = alpha * tmps[2].x + beta * Cv[2].x;
    tmps[2].y = alpha * tmps[2].y + beta * Cv[2].y;
    tmps[2].z = alpha * tmps[2].z + beta * Cv[2].z;
    tmps[2].w = alpha * tmps[2].w + beta * Cv[2].w;
    tmps[3].x = alpha * tmps[3].x + beta * Cv[3].x;
    tmps[3].y = alpha * tmps[3].y + beta * Cv[3].y;
    tmps[3].z = alpha * tmps[3].z + beta * Cv[3].z;
    tmps[3].w = alpha * tmps[3].w + beta * Cv[3].w;

    // tmps[x][1,2,3,4]对应C[x][1,2,3,4]
    *((float4*)(&C[IDX8(row_a, col_c, ldc)])) = tmps[0];
    *((float4*)(&C[IDX8(row_a, col_c+1, ldc)])) = tmps[1];
    *((float4*)(&C[IDX8(row_a, col_c+2, ldc)])) = tmps[2];
    *((float4*)(&C[IDX8(row_a, col_c+3, ldc)])) = tmps[3];
}

// 二维线程块
__global__ __launch_bounds__(256) void gemm9_4x4_micro_kernel_2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 64*64, thread size: 16*16
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    #define IDX9(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
    #define SAIDX(i,j) IDX9(i,j,(1<<6))     // shared memory a index
    #define SBIDX(i,j) IDX9(i,j,(1<<4))     // shared memory b index

    int tx_a = tx << 2, ty_a = ty;
    int tx_b = (tx&3)<<2, ty_b = ((ty<<4)+tx)>>2;
    int tx_c = tx << 2, ty_c = ty << 2;

    int lda16 = lda<<4;     // block size 64, lda * 64

    A = &(A[(bx << 6)]);
    B = &(B[(by << 6) * ldb]);
    C = &(C[(bx << 6) + (by << 6) * ldc]);  // the TB size is 64.

    __shared__ float a_tile[bM8*bK8];
    __shared__ float b_tile[bK8*bN8];

    float4 Av, Bv, Cv[4], tmps[4];
    memset(tmps, 0, sizeof(tmps));

    // 1. 数据布局如何变化: from global memory to shared memory
    // 2. float4加载连续内存如何处理
    // 3. 数据在shared memory中的位置以及如何计算
    for (int k_idx = 0; k_idx < K; k_idx += bK8) {
        Av = *((float4 *)(&A[IDX8(tx_a, ty_a, lda)]));   // 从A中取连续的4个数
        Bv = *((float4 *)(&B[IDX8(tx_b, ty_b, ldb)]));   // 从B中取连续的4个数
        ((float4 *)a_tile)[tx+ty*bx] = Av;            // 将Av中的4个数存入共享内存中的连续内存
        // ((float4 *)b_tile)[tx+ty*16] = Bv;
        b_tile[SAIDX(ty_b, tx_b)] = Bv.x;     // 将Bv中的4个数存入共享内存, 内存不连续, 之间的距离为64
        b_tile[SAIDX(ty_b, tx_b+1)] = Bv.y;
        b_tile[SAIDX(ty_b, tx_b+2)] = Bv.z;
        b_tile[SAIDX(ty_b, tx_b+3)] = Bv.w;
        A += lda16; // A偏移64列 64*lda (列优先)
        B += bK8;   // B偏移16行
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK8; ++inner_k) {   // bK8=16, 数据宽度为64, 每次计算4*4个数
            Av = *((float4 *)(&a_tile[SAIDX(tx_a, inner_k)])); // 从共享内存中取出连续的4个值给Av，需要和b相乘并累加
            Bv = *((float4 *)(&b_tile[SAIDX(ty_c, inner_k)])); // 从共享内存中取出连续的4个值给Bv，这4个数对应bK8中每个维度的值

            tmps[0].x += Av.x * Bv.x;
            tmps[0].y += Av.y * Bv.x;
            tmps[0].z += Av.z * Bv.x;
            tmps[0].w += Av.w * Bv.x;
            tmps[1].x += Av.x * Bv.y;
            tmps[1].y += Av.y * Bv.y;
            tmps[1].z += Av.z * Bv.y;
            tmps[1].w += Av.w * Bv.y;
            tmps[2].x += Av.x * Bv.z;
            tmps[2].y += Av.y * Bv.z;
            tmps[2].z += Av.z * Bv.z;
            tmps[2].w += Av.w * Bv.z;
            tmps[3].x += Av.x * Bv.w;
            tmps[3].y += Av.y * Bv.w;
            tmps[3].z += Av.z * Bv.w;
            tmps[3].w += Av.w * Bv.w;
        }
        __syncthreads();
    }
    Cv[0] = *((float4*)(&C[IDX9(tx_a, ty_c, ldc)]));
    Cv[1] = *((float4*)(&C[IDX9(tx_a, ty_c+1, ldc)]));
    Cv[2] = *((float4*)(&C[IDX9(tx_a, ty_c+2, ldc)]));
    Cv[3] = *((float4*)(&C[IDX9(tx_a, ty_c+3, ldc)]));

    tmps[0].x = alpha * tmps[0].x + beta * Cv[0].x;
    tmps[0].y = alpha * tmps[0].y + beta * Cv[0].y;
    tmps[0].z = alpha * tmps[0].z + beta * Cv[0].z;
    tmps[0].w = alpha * tmps[0].w + beta * Cv[0].w;
    tmps[1].x = alpha * tmps[1].x + beta * Cv[1].x;
    tmps[1].y = alpha * tmps[1].y + beta * Cv[1].y;
    tmps[1].z = alpha * tmps[1].z + beta * Cv[1].z;
    tmps[1].w = alpha * tmps[1].w + beta * Cv[1].w;
    tmps[2].x = alpha * tmps[2].x + beta * Cv[2].x;
    tmps[2].y = alpha * tmps[2].y + beta * Cv[2].y;
    tmps[2].z = alpha * tmps[2].z + beta * Cv[2].z;
    tmps[2].w = alpha * tmps[2].w + beta * Cv[2].w;
    tmps[3].x = alpha * tmps[3].x + beta * Cv[3].x;
    tmps[3].y = alpha * tmps[3].y + beta * Cv[3].y;
    tmps[3].z = alpha * tmps[3].z + beta * Cv[3].z;
    tmps[3].w = alpha * tmps[3].w + beta * Cv[3].w;

    *((float4*)(&C[IDX9(tx_a, ty_c, ldc)])) = tmps[0];
    *((float4*)(&C[IDX9(tx_a, ty_c+1, ldc)])) = tmps[1];
    *((float4*)(&C[IDX9(tx_a, ty_c+2, ldc)])) = tmps[2];
    *((float4*)(&C[IDX9(tx_a, ty_c+3, ldc)])) = tmps[3];
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
    gemm1_naive<<<grid, block>>>(M, N, K, dA, dB, dC);
    gemm2_tile<<<grid, block>>>(M, N, K, dA, dB, dC);
    gemm3_tile<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm4_reduce_bank_conflict<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm5_reduce_bank_conflict<<<grid, 32*32>>>(M, N, K, dA, dB, dC);
    gemm6_4x1_micro_kernel<<<grid, 16*16>>>(M, N, K, dA, dB, dC);   // 每个线程处理4个元素
    gemm7_4x1_vectorized_load_store<<<grid, 16*16>>>(M, N, K, dA, dB, dC);   // 每个线程处理4个元素
    gemm8_4x4_micro_kernel<<<grid2, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4x4个元素
    gemm9_4x4_micro_kernel_2<<<grid2, block9>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4x4个元素
    // mysgemm_v2<<<grid, block>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    // mysgemm_v4<<<grid, 32*32>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    // mysgemm_v5<<<grid, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4个元素
    // mysgemm_v7<<<grid2, 16*16>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);   // 每个线程处理4x4个元素
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
