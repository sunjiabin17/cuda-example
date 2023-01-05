#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa5(i,j) sa5[((j)<<5) + (i)]
#define sb5(i,j) sb5[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
__global__  __launch_bounds__(256)
void mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3, col = tx>>3;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa5[MS*KS];
    __shared__ float sb5[KS*NS];
    float Cres[4] = {0., 0., 0., 0.};
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa5(row1,col)=A(row1,col);
        sa5(row2,col)=A(row2,col);
        sa5(row3,col)=A(row3,col);
        sa5(row4,col)=A(row4,col);
        sb5(col,row1)=B(row1,col);
        sb5(col,row2)=B(row2,col);
        sb5(col,row3)=B(row3,col);
        sb5(col,row4)=B(row4,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            b00 = sb5(col,inner_k_count);
            Cres[0] += sa5(row1,inner_k_count) * b00;
            Cres[1] += sa5(row2,inner_k_count) * b00;
            Cres[2] += sa5(row3,inner_k_count) * b00;
            Cres[3] += sa5(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    C(row1,col) = alpha * Cres[0] + beta*C(row1,col);
    C(row2,col) = alpha * Cres[1] + beta*C(row2,col);
    C(row3,col) = alpha * Cres[2] + beta*C(row3,col);
    C(row4,col) = alpha * Cres[3] + beta*C(row4,col);
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa7(i,j) sa7[((j)<<6) + (i)]
#define sb7(i,j) sb7[((j)<<6) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 16
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v7(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row_a = (tx&15)<<2, col_a = tx>>4;
    int row_b = (tx&3)<<2, col_b = tx>>2;
    int col_c = col_a<<2;
    int lda16 = lda<<4;
    A = &A((bx<<6),0);
    B = &B(0,(by<<6));
    C = &C((bx<<6),(by<<6));//the TB size is 64.
    __shared__ float sa7[1024];
    __shared__ float sb7[1024];
    float4 Av, Bv, Cv[4], Cres[4];
    memset(Cres, 0, sizeof(Cres));
    for (int k_count = 0; k_count<K; k_count+=KS_7){
        vload(Av, &A(row_a,col_a))
        vload(Bv, &B(row_b,col_b))
        ((float4 *)sa7)[tx] = Av;
        sb7(col_b,row_b)=Bv.x;
        sb7(col_b,row_b+1)=Bv.y;
        sb7(col_b,row_b+2)=Bv.z;
        sb7(col_b,row_b+3)=Bv.w;
        A+=lda16;B+=16;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_7;inner_k_count++){
            vload(Av, &sa7(row_a,inner_k_count))
            vload(Bv, &sb7(col_c,inner_k_count))
            vscal(Cres[0], Av, Bv.x)
            vscal(Cres[1], Av, Bv.y)
            vscal(Cres[2], Av, Bv.z)
            vscal(Cres[3], Av, Bv.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_a,col_c))
    vload(Cv[1], &C(row_a,col_c+1))
    vload(Cv[2], &C(row_a,col_c+2))
    vload(Cv[3], &C(row_a,col_c+3))
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    vstore(&C(row_a,col_c), Cres[0])
    vstore(&C(row_a,col_c+1), Cres[1])
    vstore(&C(row_a,col_c+2), Cres[2])
    vstore(&C(row_a,col_c+3), Cres[3])
}

