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

__global__ void reduce(int *input, int *output) {
    __shared__ int shared[SHARED_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[threadIdx.x] = input[tid];
    __syncthreads();

    // 1   2   3   4   5   6   7   8
    // |  /    |  /    |  /    |  /
    // 3       7       11      15
    // |      /        |      /
    // 10              26
    // |              /
    // 36
    for (int i = 1; i < blockDim.x; i *= 2) {
        // divergent branch
        if (threadIdx.x % (2 * i) == 0) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
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