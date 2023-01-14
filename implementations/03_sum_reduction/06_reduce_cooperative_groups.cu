#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace cooperative_groups;

using std::cout;
using std::endl;
using std::vector;
using std::accumulate;
using std::generate;


__device__ int thread_sum(int *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) {
        int4 input4 = reinterpret_cast<int4*>(input)[i];
        sum += input4.x + input4.y + input4.z + input4.w;
    }
    return sum;
}

// Reduces a thread group to a single element
__device__ int reduce_sum_thread_group(thread_group g, int *shared, int val) {
    int lane = g.thread_rank();

    for (int i = g.size() / 2; i > 0; i >>= 1) {
        shared[lane] = val;
        g.sync();
        if (lane < i) {
            val += shared[lane + i];
        }
        g.sync();
    }
    return val;
}

__global__ void reduce(int *input, int *output, int n) {
    int my_sum = thread_sum(input, n);
    extern __shared__ int shared[];
    auto g = this_thread_block();
    int block_sum = reduce_sum_thread_group(g, shared, my_sum);
    if (g.thread_rank() == 0) {
        atomicAdd(output, block_sum);
    }
}

int main() {
    constexpr int N = 1 << 13;
    int size = N * sizeof(int);
    vector<int> h_input(N);
    vector<int> h_output(N);

    generate(h_input.begin(), h_input.end(), []() { return rand() % 100; });
    int h_sum = accumulate(h_input.begin(), h_input.end(), 0);

    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    
    constexpr int THREADS = 256;
    constexpr int BLOCKS = (N + THREADS - 1) / THREADS;

    reduce<<<BLOCKS, THREADS, N * sizeof(int)>>>(d_input, d_output, N);

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);


    cout << "Host sum: " << h_sum << endl;
    cout << "Device sum: " << h_output[0] << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}