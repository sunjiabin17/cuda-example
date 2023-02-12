#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#define Ms128 128
#define Ns128 128
#define Ks8 8

#define Mr8 8
#define Nr8 8

#define Mw 32
#define Nw 64
// Mw =4×Mr =32, Nw =8×Nr =64

#define vscal(a, b, c)\
    a.x += b * c.x;\
    a.y += b * c.y;\
    a.z += b * c.z;\
    a.w += b * c.w;

#define smemIDX(i, j, ld) ((i) * (ld) + (j))

// 一个线程计算8x8, warp tile
// ({Ms128,Ns128,Ks8}={128,128,8}, {Mr8,Nr8}={8,8})
__global__ void sgemm9(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
                      float beta, float *C, int ldc) {
    int tx = threadIdx.x & 15;       // 线程索引
    int bx = blockIdx.x, by = blockIdx.y;   // 线程块索引
    int ty = threadIdx.x >> 4;
    
    int col_a = (tx&1) << 2;        // 0 4 0 4...
    int row_a = (ty<<3) + (tx>>1);

    int row_b = ty >> 1;
    int col_b = (tx << 2) + (ty&1) * (64);
    // int row_c = tx << 3;
    // int col_c = ty << 3;
    // 从global memory加载到shared memory的方式不变
    // warp tile 如何计算row_c, col_c

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int col_w = (lane_id / 2) % 8;
    int row_w = (lane_id / 16) * 2 + (lane_id % 2);

    // int row_c = (warp_id/2) * Mw + row_w * 8;
    // int col_c = (warp_id%2) * Nw + col_w * 8;
    int row_c = (warp_id%2) * Nw + col_w * 8;
    int col_c = (warp_id/2) * Mw + row_w * 8;

    float* pA = &(A[bx * Ms128 * lda]);
    float* pB = &(B[by * Ns128]);
    float* pC = &(C[bx * Ms128 * ldc + by * Ns128]);

    // __shared__ float a_shared[Ms128][Ks8];
    __shared__ float a_shared[2][Ks8*Ms128];  // here, transpose a_shared after loaded from global to shared
    __shared__ float b_shared[2][Ks8*Ns128];
    float *ptr_sa, *ptr_sb;
    ptr_sa = (float *)a_shared;
    ptr_sb = (float *)b_shared;
    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cv[16], sums[16];
    float4 pref_Av, pref_Bv;
    memset(sums, 0.f, sizeof(sums));
    
    // 1. 加载第一个Ks大小块到寄存器, vectorized load
    pref_Av = *((float4 *)(&pA[row_a * lda + col_a]));   // load 4 consecutive elements from A
    pref_Bv = *((float4 *)(&pB[row_b * ldb + col_b]));
    // 存入共享内存
    ptr_sa[smemIDX(col_a, row_a, Ms128)] = pref_Av.x;                  // transpose A
    ptr_sa[smemIDX(col_a+1, row_a, Ms128)] = pref_Av.y;
    ptr_sa[smemIDX(col_a+2, row_a, Ms128)] = pref_Av.z;
    ptr_sa[smemIDX(col_a+3, row_a, Ms128)] = pref_Av.w;
    *((float4 *)(&ptr_sb[smemIDX(row_b, col_b, Ns128)])) = pref_Bv;
    __syncthreads();

    // 2. 加载第一个A的8个元素和B的8个元素到寄存器Av[0] Bv[0], 准备做向量外积
    Av1[0] = *((float4 *)(&ptr_sa[smemIDX(0, row_c, Ms128)]));
    Av2[0] = *((float4 *)(&ptr_sa[smemIDX(0, row_c+4, Ms128)]));
    Bv1[0] = *((float4 *)(&ptr_sb[smemIDX(0, col_c, Ns128)]));
    Bv2[0] = *((float4 *)(&ptr_sb[smemIDX(0, col_c+4, Ns128)]));

    for (int k_idx = 0; k_idx < K; k_idx += Ks8) {
        // 已经加载了第一个Ks块, 提前做偏移
        pA = pA + Ks8;
        pB = pB + Ks8 * ldb;
        int offset = (((k_idx/Ks8)+1)&1)<<10; // 切换到共享内存的第一行还是第二行
        // 3. 加载下一个Ks大小块
        pref_Av = *((float4 *)(&pA[row_a * lda + col_a]));
        pref_Bv = *((float4 *)(&pB[row_b * ldb + col_b]));
        // 外积计算并累加
        /**
         * 由于从global memory中取数的时钟周期非常多。
         * 所以在等待数据取回的同时，对read SM中的数据进行计算。
         * 也就是我们在等待的同时，需要开启8次小迭代来进行计算。
         * 而小迭代中也存在着读写分离，在对read REG进行计算之前，需要先执行write REG的操作，通过这种方式来掩盖访存的latency。
         */

        #pragma unroll
        for (int kk = 0; kk < Ks8; ++kk) {
            // 4. 预加载下一次小迭代A的8个元素和B的8个元素到寄存器Av[1] Bv[1]
            Av1[(kk+1)&1] = *((float4 *)(&ptr_sa[smemIDX((kk+1)%Ks8, row_c, Ms128)])); // A中的4个连续的数分别对应C的每一行
            Av2[(kk+1)&1] = *((float4 *)(&ptr_sa[smemIDX((kk+1)%Ks8, row_c+4, Ms128)]));
            Bv1[(kk+1)&1] = *((float4 *)(&ptr_sb[smemIDX((kk+1)%Ks8, col_c, Ns128)]));
            Bv2[(kk+1)&1] = *((float4 *)(&ptr_sb[smemIDX((kk+1)%Ks8, col_c+4, Ns128)]));
            vscal(sums[0], Av1[(kk&1)].x, Bv1[(kk&1)])
            vscal(sums[1], Av1[(kk&1)].x, Bv2[(kk&1)])
            vscal(sums[2], Av1[(kk&1)].y, Bv1[(kk&1)])
            vscal(sums[3], Av1[(kk&1)].y, Bv2[(kk&1)])
            vscal(sums[4], Av1[(kk&1)].z, Bv1[(kk&1)])
            vscal(sums[5], Av1[(kk&1)].z, Bv2[(kk&1)])
            vscal(sums[6], Av1[(kk&1)].w, Bv1[(kk&1)])
            vscal(sums[7], Av1[(kk&1)].w, Bv2[(kk&1)])
            vscal(sums[8], Av2[(kk&1)].x, Bv1[(kk&1)])
            vscal(sums[9], Av2[(kk&1)].x, Bv2[(kk&1)])
            vscal(sums[10], Av2[(kk&1)].y, Bv1[(kk&1)])
            vscal(sums[11], Av2[(kk&1)].y, Bv2[(kk&1)])
            vscal(sums[12], Av2[(kk&1)].z, Bv1[(kk&1)])
            vscal(sums[13], Av2[(kk&1)].z, Bv2[(kk&1)])
            vscal(sums[14], Av2[(kk&1)].w, Bv1[(kk&1)])
            vscal(sums[15], Av2[(kk&1)].w, Bv2[(kk&1)])
        }   
        // 5. 将 3 中加载的A和B的16个元素存入另一块共享内存并同步
        ptr_sa = (float *)a_shared + offset;
        ptr_sb = (float *)b_shared + offset;
        ptr_sa[smemIDX(col_a, row_a, Ms128)] = pref_Av.x;
        ptr_sa[smemIDX(col_a+1, row_a, Ms128)] = pref_Av.y;
        ptr_sa[smemIDX(col_a+2, row_a, Ms128)] = pref_Av.z;
        ptr_sa[smemIDX(col_a+3, row_a, Ms128)] = pref_Av.w;
        *((float4 *)(&ptr_sb[smemIDX(row_b, col_b, Ns128)])) = pref_Bv;
        __syncthreads();
        // 6. 预加载下一次大循环开始的A的8个元素和B的8个元素到寄存器Av[0] Bv[0]
        Av1[0] = *((float4 *)(&ptr_sa[smemIDX(0, row_c, Ms128)]));
        Av2[0] = *((float4 *)(&ptr_sa[smemIDX(0, row_c+4, Ms128)]));
        Bv1[0] = *((float4 *)(&ptr_sb[smemIDX(0, col_c, Ns128)]));
        Bv2[0] = *((float4 *)(&ptr_sb[smemIDX(0, col_c+4, Ns128)]));
    }
    // Cv[0] = *((float4 *)(&pC[row_c*ldc + col_c]));
    // Cv[1] = *((float4 *)(&pC[(row_c+1)*ldc + col_c]));
    // Cv[2] = *((float4 *)(&pC[(row_c+2)*ldc + col_c]));
    // Cv[3] = *((float4 *)(&pC[(row_c+3)*ldc + col_c]));

    // sums[0].x = alpha * sums[0].x + beta * Cv[0].x;

    *((float4 *)(&pC[row_c*ldc + col_c])) = sums[0];
    *((float4 *)(&pC[row_c*ldc + (col_c+4)])) = sums[1];
    *((float4 *)(&pC[(row_c+1)*ldc + col_c])) = sums[2];
    *((float4 *)(&pC[(row_c+1)*ldc + col_c+4])) = sums[3];
    *((float4 *)(&pC[(row_c+2)*ldc + col_c])) = sums[4];
    *((float4 *)(&pC[(row_c+2)*ldc + col_c+4])) = sums[5];
    *((float4 *)(&pC[(row_c+3)*ldc + col_c])) = sums[6];
    *((float4 *)(&pC[(row_c+3)*ldc + col_c+4])) = sums[7];
    *((float4 *)(&pC[(row_c+4)*ldc + col_c])) = sums[8];
    *((float4 *)(&pC[(row_c+4)*ldc + col_c+4])) = sums[9];
    *((float4 *)(&pC[(row_c+5)*ldc + col_c])) = sums[10];
    *((float4 *)(&pC[(row_c+5)*ldc + col_c+4])) = sums[11];
    *((float4 *)(&pC[(row_c+6)*ldc + col_c])) = sums[12];
    *((float4 *)(&pC[(row_c+6)*ldc + col_c+4])) = sums[13];
    *((float4 *)(&pC[(row_c+7)*ldc + col_c])) = sums[14];
    *((float4 *)(&pC[(row_c+7)*ldc + col_c+4])) = sums[15];
}


void test_sgemm9(int M, int N, int K, float *alpha, float *dA, int lda, float *dB, int ldb, 
                float *beta, float *dC, int ldc) {
    dim3 block(16*16);
    dim3 grid((M + Ms128 - 1) / Ms128, (N + Ns128 - 1) / Ns128);
    
    sgemm9<<<grid, block>>>(M, N, K, *alpha, dA, lda, dB, ldb, *beta, dC, ldc);
    cudaDeviceSynchronize();
    
}