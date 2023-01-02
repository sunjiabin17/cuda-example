#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using std::cout;
using std::endl;
using std::vector;
using std::accumulate;
using std::generate;

constexpr int THREADS = 256;
constexpr int SHARED_SIZE = 256;

// 解决warp divergence: 使同一个warp里的threads走同一个分支
__global__ void reduce(int *input, int *output) {
    __shared__ int shared[SHARED_SIZE];
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[threadIdx.x] = input[gid];
    __syncthreads();

    // 0   1   2   3   4   5   6   7
    // 第一次迭代: i = 1
    // thread 0: index = 0, 计算 shared[0]+shared[1]，保存在shared[0]
    // thread 1: index = 2, 计算 shared[2]+shared[3]，保存在shared[2]
    // thread 2: index = 4, 计算 shared[4]+shared[5]，保存在shared[4]
    // thread 3: index = 6, 计算 shared[6]+shared[7]，保存在shared[6]
    // thread 4-7: index >= blockDim.x，不执行
    // 第二次迭代: i = 2
    // thread 0: index = 0, 计算 shared[0]+shared[2]，保存在shared[0]
    // thread 1: index = 4, 计算 shared[4]+shared[6]，保存在shared[4]
    // thread 2-7: index >= blockDim.x，不执行
    // 第三次迭代: i = 4
    // thread 0: index = 0, 计算 shared[0]+shared[4]，保存在shared[0]
    // thread 1-7: index >= blockDim.x，不执行
    // 第四次迭代: i = 8 超出blockDim.x，退出循环
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * threadIdx.x;
        if (index < blockDim.x) {
            shared[index] += shared[index + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared[0];
    }
}

int main() {
    constexpr int N = 1 << 16;
    int size = N * sizeof(int);
    vector<int> h_input(N);
    vector<int> h_output(N);

    generate(h_input.begin(), h_input.end(), []() { return rand() % 100; });
    int h_sum = accumulate(h_input.begin(), h_input.end(), 0);

    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    constexpr int BLOCKS = N / THREADS;

    reduce<<<BLOCKS, THREADS>>>(d_input, d_output);

    reduce<<<1, BLOCKS>>>(d_output, d_output);

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);


    cout << "Host sum: " << h_sum << endl;
    cout << "Device sum: " << h_output[0] << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}