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
// ({Ms,Ns,Ks}={64,64,16}, {Mr,Nr}={4,4})
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
    int lda16 = lda<<4;

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
        A += lda16; // A偏移16列
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
__global__ __launch_bounds__(256) void gemm8_4x4_micro_kernel_2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
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

    int lda16 = lda<<4;

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
        A += lda16; // A偏移16列 16*lda
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


//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;

#define vfma(v1, alpha, v2, beta, v3)\
    v1.x = alpha * v2.x + beta * v3.x;\
    v1.y = alpha * v2.y + beta * v3.y;\
    v1.z = alpha * v2.z + beta * v3.z;\
    v1.w = alpha * v2.w + beta * v3.w;

#define bM9 128
#define bN9 128
#define bK9 8
// ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8})
__global__ __launch_bounds__(256) void gemm9_8x8_micro_kernel(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 128*128, thread size: 16*16
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim
    int bx = blockIdx.x, by = blockIdx.y;
    
    #define IDX9(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
    #define SIDX9(i,j) IDX9(i,j,(1<<7))     // shared memory index

    int row_a = (tx&31)<<2, col_a = tx>>5;  // row_a = range(0, 128, 4), col_a = range(0, 8, 1)
    int row_b = (tx&1)<<2, col_b = tx>>1;   // row_b = range(0, 8, 4), col_b = range(0, 128, 1)
    int row_c = (tx&15)<<3, col_c = (tx>>4)<<3; // row_c = range(0, 128, 8), col_c = range(0, 128, 8)
    int lda8 = lda<<3;

    A = &(A[(bx << 7)]);
    B = &(B[(by << 7) * ldb]);
    C = &(C[(bx << 7) + (by << 7) * ldc]);  // the TB size is 128.

    __shared__ float a_tile[bM9*bK9];
    __shared__ float b_tile[bK9*bN9];

    float4 Av1, Av2, Bv1, Bv2, Cv[16], tmps[16];
    memset(tmps, 0, sizeof(tmps));

    for (int k_idx = 0; k_idx < K; k_idx += bK9) {
        // 从A,B中取连续的4个数, 每个线程从A中取4个数, 从B中取4个数
        Av1 = *((float4 *)(&A[IDX9(row_a, col_a, lda)]));
        Bv1 = *((float4 *)(&B[IDX9(row_b, col_b, ldb)]));
        // 存到共享内存中, 每个线程存4个数, 存储位置连续
        ((float4 *)a_tile)[tx] = Av1;
        // ((float4 *)b_tile)[tx] = Bv;
        // 存在共享内存中, 每个线程存4个数, 但是存储位置不连续, 之间的距离为128
        b_tile[SIDX9(col_b, row_b)] = Bv1.x;
        b_tile[SIDX9(col_b, row_b+1)] = Bv1.y;
        b_tile[SIDX9(col_b, row_b+2)] = Bv1.z;
        b_tile[SIDX9(col_b, row_b+3)] = Bv1.w;
        A += lda8; // A偏移8列 8*lda (列优先)
        B += bK9;   // B偏移8行
        __syncthreads();
        
        #pragma unroll
        // 每个线程每次循环从共享内存A中读取8个数, 从共享内存B中读取8个数，然后进行8*8次累加计算; 共8bK9次循环
        for (int inner_k = 0; inner_k < bK9; ++inner_k) {   // bK9=8, 数据宽度为128
            Av1 = *((float4 *)(&a_tile[SIDX9(row_c, inner_k)])); // 从共享内存中取出连续的4个值给Av，需要和b相乘并累加
            Av2 = *((float4 *)(&a_tile[SIDX9(row_c+4, inner_k)]));
            Bv1 = *((float4 *)(&b_tile[SIDX9(col_c, inner_k)])); // 从共享内存中取出连续的4个值给Bv，这4个数对应bK8中每个维度的值
            Bv2 = *((float4 *)(&b_tile[SIDX9(col_c+4, inner_k)]));
            vscal(tmps[0], Av1, Bv1.x)     // tmps[x][y] 对应计算A[x][:] * B[:][y], 结果存在C[x][y]处
            vscal(tmps[1], Av2, Bv1.x)
            vscal(tmps[2], Av1, Bv1.y)
            vscal(tmps[3], Av2, Bv1.y)
            vscal(tmps[4], Av1, Bv1.z)
            vscal(tmps[5], Av2, Bv1.z)
            vscal(tmps[6], Av1, Bv1.w)
            vscal(tmps[7], Av2, Bv1.w)
            vscal(tmps[8], Av1, Bv2.x)
            vscal(tmps[9], Av2, Bv2.x)
            vscal(tmps[10], Av1, Bv2.y)
            vscal(tmps[11], Av2, Bv2.y)
            vscal(tmps[12], Av1, Bv2.z)
            vscal(tmps[13], Av2, Bv2.z)
            vscal(tmps[14], Av1, Bv2.w)
            vscal(tmps[15], Av2, Bv2.w)
        }
        __syncthreads();
    }
    // 每个线程从全局内存中读取8*8个数
    // 从全局内存中取出row_a, col_c处连续的4个值给Cv
    // row_c = range(0, 128, 8), col_c = range(0, 128, 8)
    Cv[0] = *((float4*)(&C[IDX9(row_c, col_c, ldc)]));
    Cv[1] = *((float4*)(&C[IDX9(row_c+4, col_c, ldc)]));
    Cv[2] = *((float4*)(&C[IDX9(row_c, col_c+1, ldc)]));
    Cv[3] = *((float4*)(&C[IDX9(row_c+4, col_c+1, ldc)]));
    Cv[4] = *((float4*)(&C[IDX9(row_c, col_c+2, ldc)]));
    Cv[5] = *((float4*)(&C[IDX9(row_c+4, col_c+2, ldc)]));
    Cv[6] = *((float4*)(&C[IDX9(row_c, col_c+3, ldc)]));
    Cv[7] = *((float4*)(&C[IDX9(row_c+4, col_c+3, ldc)]));
    Cv[8] = *((float4*)(&C[IDX9(row_c, col_c+4, ldc)]));
    Cv[9] = *((float4*)(&C[IDX9(row_c+4, col_c+4, ldc)]));
    Cv[10] = *((float4*)(&C[IDX9(row_c, col_c+5, ldc)]));
    Cv[11] = *((float4*)(&C[IDX9(row_c+4, col_c+5, ldc)]));
    Cv[12] = *((float4*)(&C[IDX9(row_c, col_c+6, ldc)]));
    Cv[13] = *((float4*)(&C[IDX9(row_c+4, col_c+6, ldc)]));
    Cv[14] = *((float4*)(&C[IDX9(row_c, col_c+7, ldc)]));
    Cv[15] = *((float4*)(&C[IDX9(row_c+4, col_c+7, ldc)]));
    
    // 每个线程进行8*8次计算
    vfma(tmps[0], alpha, tmps[0], beta, Cv[0])
    vfma(tmps[1], alpha, tmps[1], beta, Cv[1])
    vfma(tmps[2], alpha, tmps[2], beta, Cv[2])
    vfma(tmps[3], alpha, tmps[3], beta, Cv[3])
    vfma(tmps[4], alpha, tmps[4], beta, Cv[4])
    vfma(tmps[5], alpha, tmps[5], beta, Cv[5])
    vfma(tmps[6], alpha, tmps[6], beta, Cv[6])
    vfma(tmps[7], alpha, tmps[7], beta, Cv[7])
    vfma(tmps[8], alpha, tmps[8], beta, Cv[8])
    vfma(tmps[9], alpha, tmps[9], beta, Cv[9])
    vfma(tmps[10], alpha, tmps[10], beta, Cv[10])
    vfma(tmps[11], alpha, tmps[11], beta, Cv[11])
    vfma(tmps[12], alpha, tmps[12], beta, Cv[12])
    vfma(tmps[13], alpha, tmps[13], beta, Cv[13])
    vfma(tmps[14], alpha, tmps[14], beta, Cv[14])
    vfma(tmps[15], alpha, tmps[15], beta, Cv[15])

    // 每个线程写入8*8个数到全局内存
    // 将tmps中的值写回到C中
    *((float4*)(&C[IDX9(row_c, col_c, ldc)])) = tmps[0];
    *((float4*)(&C[IDX9(row_c+4, col_c, ldc)])) = tmps[1];
    *((float4*)(&C[IDX9(row_c, col_c+1, ldc)])) = tmps[2];
    *((float4*)(&C[IDX9(row_c+4, col_c+1, ldc)])) = tmps[3];
    *((float4*)(&C[IDX9(row_c, col_c+2, ldc)])) = tmps[4];
    *((float4*)(&C[IDX9(row_c+4, col_c+2, ldc)])) = tmps[5];
    *((float4*)(&C[IDX9(row_c, col_c+3, ldc)])) = tmps[6];
    *((float4*)(&C[IDX9(row_c+4, col_c+3, ldc)])) = tmps[7];
    *((float4*)(&C[IDX9(row_c, col_c+4, ldc)])) = tmps[8];
    *((float4*)(&C[IDX9(row_c+4, col_c+4, ldc)])) = tmps[9];
    *((float4*)(&C[IDX9(row_c, col_c+5, ldc)])) = tmps[10];
    *((float4*)(&C[IDX9(row_c+4, col_c+5, ldc)])) = tmps[11];
    *((float4*)(&C[IDX9(row_c, col_c+6, ldc)])) = tmps[12];
    *((float4*)(&C[IDX9(row_c+4, col_c+6, ldc)])) = tmps[13];
    *((float4*)(&C[IDX9(row_c, col_c+7, ldc)])) = tmps[14];
    *((float4*)(&C[IDX9(row_c+4, col_c+7, ldc)])) = tmps[15];
}


#define bM10 128
#define bN10 128
#define bK10 8

#define IDX10(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX10(i,j) IDX10(i,j,(1<<7))     // shared memory index

// ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8})
__global__ __launch_bounds__(256) void gemm10_8x8_warp_tile(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 128*128, thread size: 16*16, 每个block有8个warp
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;           // tx = 0-255
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx >> 5;              // tx/32, warp_id = 0-7
    int lane_id = tx & 31;              // tx%32, lane_id = 0-31
    int warp_row = warp_id & 3;         // warp_id%4, warp_row = 0-3
    int warp_col = warp_id >> 2;        // warp_id/4, warp_col = 0-1
    int row_w = lane_id & 3;            // lane_id%4, row_w = 0-3
    int col_w = lane_id >> 2;           // lane_id/4, col_w = 0-7

    int row_a = (tx&31)<<2, col_a = tx>>5;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (warp_row<<5) + (row_w<<3);
    int col_c = (warp_col<<6) + (col_w<<3);


    A = &(A[(bx<<7)]);
    B = &(B[(by<<7) * ldb]);
    C = &(C[(bx<<7) + (by<<7) * ldc]);

    __shared__ float a_tile[bM10*bK10];
    __shared__ float b_tile[bK10*bN10];

    float4 Av1, Av2, Bv1, Bv2, Cv[16], tmps[16];
    memset(tmps, 0, sizeof(tmps));

    for (int k_idx = 0; k_idx < K; k_idx += bK10) {
        Av1 = *((float4 *)(&A[IDX10(row_a, col_a, lda)]));
        Bv1 = *((float4 *)(&B[IDX10(row_b, col_b, ldb)]));
        ((float4 *)a_tile)[tx] = Av1;
        b_tile[SIDX10(col_b, row_b)] = Bv1.x;
        b_tile[SIDX10(col_b, row_b+1)] = Bv1.y;
        b_tile[SIDX10(col_b, row_b+2)] = Bv1.z;
        b_tile[SIDX10(col_b, row_b+3)] = Bv1.w;
        A += lda8; B += bK10;
        __syncthreads();

        #pragma unroll
        for (int inner_k = 0; inner_k < bK10; ++inner_k) {
            Av1 = *((float4 *)(&a_tile[SIDX10(row_c, inner_k)]));
            Av2 = *((float4 *)(&a_tile[SIDX10(row_c+4, inner_k)]));
            Bv1 = *((float4 *)(&b_tile[SIDX10(col_c, inner_k)]));
            Bv2 = *((float4 *)(&b_tile[SIDX10(col_c+4, inner_k)]));
            vscal(tmps[0], Av1, Bv1.x)     // tmps[x][y] 对应计算A[x][:] * B[:][y], 结果存在C[x][y]处
            vscal(tmps[1], Av2, Bv1.x)
            vscal(tmps[2], Av1, Bv1.y)
            vscal(tmps[3], Av2, Bv1.y)
            vscal(tmps[4], Av1, Bv1.z)
            vscal(tmps[5], Av2, Bv1.z)
            vscal(tmps[6], Av1, Bv1.w)
            vscal(tmps[7], Av2, Bv1.w)
            vscal(tmps[8], Av1, Bv2.x)
            vscal(tmps[9], Av2, Bv2.x)
            vscal(tmps[10], Av1, Bv2.y)
            vscal(tmps[11], Av2, Bv2.y)
            vscal(tmps[12], Av1, Bv2.z)
            vscal(tmps[13], Av2, Bv2.z)
            vscal(tmps[14], Av1, Bv2.w)
            vscal(tmps[15], Av2, Bv2.w)
        }
        __syncthreads();
    }

    // 每个线程从全局内存中读取8*8个数
    // 从全局内存中取出row_a, col_c处连续的4个值给Cv
    Cv[0] = *((float4 *)(&C[IDX10(row_c, col_c, ldc)]));
    Cv[1] = *((float4 *)(&C[IDX10(row_c+4, col_c, ldc)]));
    Cv[2] = *((float4 *)(&C[IDX10(row_c, col_c+1, ldc)]));
    Cv[3] = *((float4 *)(&C[IDX10(row_c+4, col_c+1, ldc)]));
    Cv[4] = *((float4 *)(&C[IDX10(row_c, col_c+2, ldc)]));
    Cv[5] = *((float4 *)(&C[IDX10(row_c+4, col_c+2, ldc)]));
    Cv[6] = *((float4 *)(&C[IDX10(row_c, col_c+3, ldc)]));
    Cv[7] = *((float4 *)(&C[IDX10(row_c+4, col_c+3, ldc)]));
    Cv[8] = *((float4 *)(&C[IDX10(row_c, col_c+4, ldc)]));
    Cv[9] = *((float4 *)(&C[IDX10(row_c+4, col_c+4, ldc)]));
    Cv[10] = *((float4 *)(&C[IDX10(row_c, col_c+5, ldc)]));
    Cv[11] = *((float4 *)(&C[IDX10(row_c+4, col_c+5, ldc)]));
    Cv[12] = *((float4 *)(&C[IDX10(row_c, col_c+6, ldc)]));
    Cv[13] = *((float4 *)(&C[IDX10(row_c+4, col_c+6, ldc)]));
    Cv[14] = *((float4 *)(&C[IDX10(row_c, col_c+7, ldc)]));
    Cv[15] = *((float4 *)(&C[IDX10(row_c+4, col_c+7, ldc)]));

    // 每个线程进行8*8次计算
    vfma(tmps[0], alpha, tmps[0], beta, Cv[0])
    vfma(tmps[1], alpha, tmps[1], beta, Cv[1])
    vfma(tmps[2], alpha, tmps[2], beta, Cv[2])
    vfma(tmps[3], alpha, tmps[3], beta, Cv[3])
    vfma(tmps[4], alpha, tmps[4], beta, Cv[4])
    vfma(tmps[5], alpha, tmps[5], beta, Cv[5])
    vfma(tmps[6], alpha, tmps[6], beta, Cv[6])
    vfma(tmps[7], alpha, tmps[7], beta, Cv[7])
    vfma(tmps[8], alpha, tmps[8], beta, Cv[8])
    vfma(tmps[9], alpha, tmps[9], beta, Cv[9])
    vfma(tmps[10], alpha, tmps[10], beta, Cv[10])
    vfma(tmps[11], alpha, tmps[11], beta, Cv[11])
    vfma(tmps[12], alpha, tmps[12], beta, Cv[12])
    vfma(tmps[13], alpha, tmps[13], beta, Cv[13])
    vfma(tmps[14], alpha, tmps[14], beta, Cv[14])
    vfma(tmps[15], alpha, tmps[15], beta, Cv[15])

    // 每个线程写入8*8个数到全局内存
    *((float4 *)(&C[IDX10(row_c, col_c, ldc)])) = tmps[0];
    *((float4 *)(&C[IDX10(row_c+4, col_c, ldc)])) = tmps[1];
    *((float4 *)(&C[IDX10(row_c, col_c+1, ldc)])) = tmps[2];
    *((float4 *)(&C[IDX10(row_c+4, col_c+1, ldc)])) = tmps[3];
    *((float4 *)(&C[IDX10(row_c, col_c+2, ldc)])) = tmps[4];
    *((float4 *)(&C[IDX10(row_c+4, col_c+2, ldc)])) = tmps[5];
    *((float4 *)(&C[IDX10(row_c, col_c+3, ldc)])) = tmps[6];
    *((float4 *)(&C[IDX10(row_c+4, col_c+3, ldc)])) = tmps[7];
    *((float4 *)(&C[IDX10(row_c, col_c+4, ldc)])) = tmps[8];
    *((float4 *)(&C[IDX10(row_c+4, col_c+4, ldc)])) = tmps[9];
    *((float4 *)(&C[IDX10(row_c, col_c+5, ldc)])) = tmps[10];
    *((float4 *)(&C[IDX10(row_c+4, col_c+5, ldc)])) = tmps[11];
    *((float4 *)(&C[IDX10(row_c, col_c+6, ldc)])) = tmps[12];
    *((float4 *)(&C[IDX10(row_c+4, col_c+6, ldc)])) = tmps[13];
    *((float4 *)(&C[IDX10(row_c, col_c+7, ldc)])) = tmps[14];
    *((float4 *)(&C[IDX10(row_c+4, col_c+7, ldc)])) = tmps[15];
}


//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;

#define vfma(v1, alpha, v2, beta, v3)\
    v1.x = alpha * v2.x + beta * v3.x;\
    v1.y = alpha * v2.y + beta * v3.y;\
    v1.z = alpha * v2.z + beta * v3.z;\
    v1.w = alpha * v2.w + beta * v3.w;

#define bM11 128
#define bN11 128
#define bK11 8

#define IDX11(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX11(i,j) IDX11(i,j,(1<<7))     // shared memory index

// ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8}) {Mw,Nw}={4xMr,8xNr}
__global__ __launch_bounds__(256) void gemm11_8x8_prefetch(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 128*128, thread size: 16*16, 每个block有8个warp
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;           // tx = 0-255
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx >> 5;              // tx/32, warp_id = 0-7
    int lane_id = tx & 31;              // tx%32, lane_id = 0-31
    int warp_row = warp_id & 3;         // warp_id%4, warp_row = 0-3
    int warp_col = warp_id >> 2;        // warp_id/4, warp_col = 0-1
    int row_w = lane_id & 3;            // lane_id%4, row_w = 0-3
    int col_w = lane_id >> 2;           // lane_id/4, col_w = 0-7

    int row_c = (warp_row<<5) + (row_w<<3); // stride=8
    int col_c = (warp_col<<6) + (col_w<<3);

    int row_a = (tx&31)<<2, col_a = tx>>5;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    
    int K_upper = K >> 3;   // K_upper = K/8


    A = &(A[(bx<<7)]);
    B = &(B[(by<<7) * ldb]);
    C = &(C[(bx<<7) + (by<<7) * ldc]);

    __shared__ float a_tile[bM11*bK11];
    __shared__ float b_tile[bK11*bN11];

    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cv[16], tmps[16];
    float4 pref_Av, pref_Bv;
    float *ptr_A, *ptr_B;
    memset(tmps, 0, sizeof(tmps));

    // 从全局内存加载A和B中的一个block到共享内存
    pref_Av = *((float4 *)(&A[IDX11(row_a, col_a, lda)]));
    pref_Bv = *((float4 *)(&B[IDX11(row_b, col_b, ldb)]));
    ((float4 *)a_tile)[tx] = pref_Av;
    b_tile[SIDX11(col_b, row_b)] = pref_Bv.x;
    b_tile[SIDX11(col_b, row_b+1)] = pref_Bv.y;
    b_tile[SIDX11(col_b, row_b+2)] = pref_Bv.z;
    b_tile[SIDX11(col_b, row_b+3)] = pref_Bv.w;
    __syncthreads();
    Av1[0] = *((float4 *)(&a_tile[SIDX11(row_c, 0)]));
    Av2[0] = *((float4 *)(&a_tile[SIDX11(row_c+4, 0)]));
    Bv1[0] = *((float4 *)(&b_tile[SIDX11(col_c, 0)]));
    Bv2[0] = *((float4 *)(&b_tile[SIDX11(col_c+4, 0)]));

    for (int k_idx = 0; k_idx < K_upper; ++k_idx) {
        // packing A and B into shared memory
        int inc = (k_idx+1)%K_upper;
        ptr_A = A + inc*lda8;
        ptr_B = B + inc*bK11;
        // 预取全局内存中A和B中的下一个block
        pref_Av = *((float4 *)(&ptr_A[IDX11(row_a, col_a, lda)]));
        pref_Bv = *((float4 *)(&ptr_B[IDX11(row_b, col_b, ldb)]));
        #pragma unroll
        for (int inner_k = 0; inner_k < bK11; ++inner_k) {
            int next_inner_k = (inner_k+1)&7;
            // 预取共享内存中A和B的8个元素
            Av1[(inner_k+1)&1] = *((float4 *)(&a_tile[SIDX11(row_c, next_inner_k)]));
            Av2[(inner_k+1)&1] = *((float4 *)(&a_tile[SIDX11(row_c+4, next_inner_k)]));
            Bv1[(inner_k+1)&1] = *((float4 *)(&b_tile[SIDX11(col_c, next_inner_k)]));
            Bv2[(inner_k+1)&1] = *((float4 *)(&b_tile[SIDX11(col_c+4, next_inner_k)]));
            // 计算
            vscal(tmps[0], Av1[(inner_k)&1], Bv1[(inner_k)&1].x)
            vscal(tmps[1], Av2[(inner_k)&1], Bv1[(inner_k)&1].x)
            vscal(tmps[2], Av1[(inner_k)&1], Bv1[(inner_k)&1].y)
            vscal(tmps[3], Av2[(inner_k)&1], Bv1[(inner_k)&1].y)
            vscal(tmps[4], Av1[(inner_k)&1], Bv1[(inner_k)&1].z)
            vscal(tmps[5], Av2[(inner_k)&1], Bv1[(inner_k)&1].z)
            vscal(tmps[6], Av1[(inner_k)&1], Bv1[(inner_k)&1].w)
            vscal(tmps[7], Av2[(inner_k)&1], Bv1[(inner_k)&1].w)
            vscal(tmps[8], Av1[(inner_k)&1], Bv2[(inner_k)&1].x)
            vscal(tmps[9], Av2[(inner_k)&1], Bv2[(inner_k)&1].x)
            vscal(tmps[10], Av1[(inner_k)&1], Bv2[(inner_k)&1].y)
            vscal(tmps[11], Av2[(inner_k)&1], Bv2[(inner_k)&1].y)
            vscal(tmps[12], Av1[(inner_k)&1], Bv2[(inner_k)&1].z)
            vscal(tmps[13], Av2[(inner_k)&1], Bv2[(inner_k)&1].z)
            vscal(tmps[14], Av1[(inner_k)&1], Bv2[(inner_k)&1].w)
            vscal(tmps[15], Av2[(inner_k)&1], Bv2[(inner_k)&1].w)
        }
        __syncthreads();

        // 把预取的数据放到共享内存
        ((float4 *)a_tile)[tx] = pref_Av;
        b_tile[SIDX11(col_b, row_b)] = pref_Bv.x;
        b_tile[SIDX11(col_b, row_b+1)] = pref_Bv.y;
        b_tile[SIDX11(col_b, row_b+2)] = pref_Bv.z;
        b_tile[SIDX11(col_b, row_b+3)] = pref_Bv.w;
        __syncthreads();
        // 下一次循环需要的共享内存中的8个元素
        Av1[0] = *((float4 *)(&a_tile[SIDX11(row_c, 0)]));
        Av2[0] = *((float4 *)(&a_tile[SIDX11(row_c+4, 0)]));
        Bv1[0] = *((float4 *)(&b_tile[SIDX11(col_c, 0)]));
        Bv2[0] = *((float4 *)(&b_tile[SIDX11(col_c+4, 0)]));
    }
    Cv[0] = *((float4 *)(&C[IDX11(row_c, col_c, ldc)]));
    Cv[1] = *((float4 *)(&C[IDX11(row_c+4, col_c, ldc)]));
    Cv[2] = *((float4 *)(&C[IDX11(row_c, col_c+1, ldc)]));
    Cv[3] = *((float4 *)(&C[IDX11(row_c+4, col_c+1, ldc)]));
    Cv[4] = *((float4 *)(&C[IDX11(row_c, col_c+2, ldc)]));
    Cv[5] = *((float4 *)(&C[IDX11(row_c+4, col_c+2, ldc)]));
    Cv[6] = *((float4 *)(&C[IDX11(row_c, col_c+3, ldc)]));
    Cv[7] = *((float4 *)(&C[IDX11(row_c+4, col_c+3, ldc)]));
    Cv[8] = *((float4 *)(&C[IDX11(row_c, col_c+4, ldc)]));
    Cv[9] = *((float4 *)(&C[IDX11(row_c+4, col_c+4, ldc)]));
    Cv[10] = *((float4 *)(&C[IDX11(row_c, col_c+5, ldc)]));
    Cv[11] = *((float4 *)(&C[IDX11(row_c+4, col_c+5, ldc)]));
    Cv[12] = *((float4 *)(&C[IDX11(row_c, col_c+6, ldc)]));
    Cv[13] = *((float4 *)(&C[IDX11(row_c+4, col_c+6, ldc)]));
    Cv[14] = *((float4 *)(&C[IDX11(row_c, col_c+7, ldc)]));
    Cv[15] = *((float4 *)(&C[IDX11(row_c+4, col_c+7, ldc)]));

    vfma(tmps[0], alpha, tmps[0], beta, Cv[0])
    vfma(tmps[1], alpha, tmps[1], beta, Cv[1])
    vfma(tmps[2], alpha, tmps[2], beta, Cv[2])
    vfma(tmps[3], alpha, tmps[3], beta, Cv[3])

    vfma(tmps[4], alpha, tmps[4], beta, Cv[4])
    vfma(tmps[5], alpha, tmps[5], beta, Cv[5])
    vfma(tmps[6], alpha, tmps[6], beta, Cv[6])
    vfma(tmps[7], alpha, tmps[7], beta, Cv[7])

    vfma(tmps[8], alpha, tmps[8], beta, Cv[8])
    vfma(tmps[9], alpha, tmps[9], beta, Cv[9])
    vfma(tmps[10], alpha, tmps[10], beta, Cv[10])
    vfma(tmps[11], alpha, tmps[11], beta, Cv[11])

    vfma(tmps[12], alpha, tmps[12], beta, Cv[12])
    vfma(tmps[13], alpha, tmps[13], beta, Cv[13])
    vfma(tmps[14], alpha, tmps[14], beta, Cv[14])
    vfma(tmps[15], alpha, tmps[15], beta, Cv[15])

    *((float4 *)(&C[IDX11(row_c, col_c, ldc)])) = tmps[0];
    *((float4 *)(&C[IDX11(row_c+4, col_c, ldc)])) = tmps[1];
    *((float4 *)(&C[IDX11(row_c, col_c+1, ldc)])) = tmps[2];
    *((float4 *)(&C[IDX11(row_c+4, col_c+1, ldc)])) = tmps[3];
    *((float4 *)(&C[IDX11(row_c, col_c+2, ldc)])) = tmps[4];
    *((float4 *)(&C[IDX11(row_c+4, col_c+2, ldc)])) = tmps[5];
    *((float4 *)(&C[IDX11(row_c, col_c+3, ldc)])) = tmps[6];
    *((float4 *)(&C[IDX11(row_c+4, col_c+3, ldc)])) = tmps[7];
    *((float4 *)(&C[IDX11(row_c, col_c+4, ldc)])) = tmps[8];
    *((float4 *)(&C[IDX11(row_c+4, col_c+4, ldc)])) = tmps[9];
    *((float4 *)(&C[IDX11(row_c, col_c+5, ldc)])) = tmps[10];
    *((float4 *)(&C[IDX11(row_c+4, col_c+5, ldc)])) = tmps[11];
    *((float4 *)(&C[IDX11(row_c, col_c+6, ldc)])) = tmps[12];
    *((float4 *)(&C[IDX11(row_c+4, col_c+6, ldc)])) = tmps[13];
    *((float4 *)(&C[IDX11(row_c, col_c+7, ldc)])) = tmps[14];
    *((float4 *)(&C[IDX11(row_c+4, col_c+7, ldc)])) = tmps[15];

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
