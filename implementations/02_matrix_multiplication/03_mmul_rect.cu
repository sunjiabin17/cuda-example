// note: BLOCK_SIZE的x和y不一样
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>

using std::cout;
using std::endl;
using std::vector;
using std::generate;

constexpr int M = 1 << 10;
constexpr int N = 1 << 11;
constexpr int K = 1 << 12;

constexpr int THREADS = 32;
constexpr int SHARED_SIZE = THREADS * THREADS;  // 1024


// (M, K) * (K, N) = (M, N)
__global__ void matrixMultiplication(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int a_shared[SHARED_SIZE];
    __shared__ int b_shared[SHARED_SIZE];

    int c_value = 0;

    // 把整个矩阵分成多个小块，每个小块的大小为blockDim.x * blockDim.y （32*32）
    // a矩阵的列数是K，b矩阵的行数是K
    for (int i = 0; i < K; i += blockDim.x) {
        // 把a矩阵中按先row后col的顺序划分每个小块，并加载到共享内存中
        // |1 2| |3 4|
        // |5 6| |7 8|
        // -----------
        // |9 0| |1 2|
        // |3 4| |5 6|
        // a ===> a_shared
        // |1 2|  |3 4|  |9 0|  |1 2|
        // |5 6|  |7 8|  |3 4|  |5 6|

        // row * K: 表示当前小块的第一行在a矩阵中的row偏移量
        // i: 表示当前小块的第一列在a矩阵中的col偏移量
        // threadIdx.x: 表示当前线程在小块中的col偏移量
        // note: a矩阵(M, K)
        a_shared[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
        
        // 把b矩阵中按先col后row的顺序划分每个小块，并加载到共享内存中
        // |1 2| |3 4|
        // |5 6| |7 8|
        // -----------
        // |9 0| |1 2|
        // |3 4| |5 6|
        // b ===> b_shared
        // |1 2|  |9 0|  |3 4|  |1 2|
        // |5 6|  |3 4|  |7 8|  |5 6|

        // shared memory中的b矩阵是按先col后row的顺序存储的  (例如b_shared中的第二个小块对应b矩阵中的blockIdx(1, 0)的位置)
        // i * N: 表示当前小块的第一行在b矩阵中的col偏移量(blockDim.y的整数倍)
        // threadIdx.y * N: 表示当前小块的第一行在b矩阵中的row偏移量 (blockDim.x的整数倍)
        // col: 表示当前线程在当前行中的row偏移量
        
        // 举例计算b[3][1]的值, blockIdx(1, 0)
        // i = 2, N = 4, threadIdx.y = 1, row = 3, col = 1
        // i * N = 8, threadIdx.y * N = 4, col = 1
        // b[3][1] = b[8 + 4 + 1] = b[13] = 4
        // b_shared[1][1] = b_shared[3] = 4
        // note: b矩阵(K, N)
        b_shared[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        __syncthreads();

        // 计算当前小块的结果 (blockDim.x = 32)
        for (int j = 0; j < blockDim.x; ++j) {
            c_value += a_shared[threadIdx.y * blockDim.x + j] * b_shared[j * blockDim.x + threadIdx.x];
        }
    
        __syncthreads();
    }

    // c矩阵(M, N)
    c[row * N + col] = c_value;
}


// matrix multiply on the CPU
// (M, K) * (K, N) = (M, N)
void matrixMultiplicationCPU(const vector<int>& a, const vector<int>& b, vector<int>& c) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int c_value = 0;
            for (int i = 0; i < K; ++i) {
                c_value += a[row * K + i] * b[i * N + col];
            }
            c[row * N + col] = c_value;
        }
    }
}

// verify that the GPU result matches the CPU result
bool verifyResult(const vector<int>& a, const vector<int>& b) {
    for (int i = 0; i < M * N; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}


int main() {
    vector<int> h_a(M * K);
    vector<int> h_b(K * N);
    vector<int> h_c(M * N);
    vector<int> ref_c(M * N); 

    size_t bytes_a = M * K * sizeof(int);
    size_t bytes_b = K * N * sizeof(int);
    size_t bytes_c = M * N * sizeof(int);

    // initialize the matrices
    generate(h_a.begin(), h_a.end(), []() {return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() {return rand() % 100; });

    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

    // note: BLOCK_SIZE x和y的值不同
    int BLOCKS_X = N / THREADS;
    int BLOCKS_Y = M / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    matrixMultiplication<<<blocks, threads>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

    // matrixMultiplicationCPU(h_a, h_b, ref_c);

    // if (verifyResult(h_c, ref_c)) {
    //     cout << "the result is correct" << endl;
    // } else {
    //     cout << "the result is incorrect" << endl;
    // }

    // // print some value randomly
    // for (int i = 0; i < 10; ++i) {
    //     int row = rand() % N;
    //     int col = rand() % N;
    //     cout << "h_c[" << row << "][" << col << "] = " << h_c[row * N + col] << endl;
    //     cout << "ref_c[" << row << "][" << col << "] = " << ref_c[row * N + col] << endl;
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}