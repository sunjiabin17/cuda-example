#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::vector;

constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);   // == ceil(26 / float(BINS))

__global__ void histogram(char *input, int *outout, int N) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the bin positions where threads are grouped together
    int alpha_position;
    for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
        // Calculate the position in the alphabet
        alpha_position = input[i] - 'a';
        atomicAdd(&outout[alpha_position / DIV], 1);
    }
}

int main() {
    int N = 1 << 24;
    vector<char> h_input(N);
    vector<int> h_result(BINS);

    generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

    char *d_input;
    int *d_result;
    cudaMalloc(&d_input, N);
    cudaMalloc(&d_result, BINS * sizeof(int));

    cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int), cudaMemcpyHostToDevice);

    int THREADS = 512;
    int BLOCKS = N / THREADS;

    histogram<<<BLOCKS, THREADS>>>(d_input, d_result, N);

    cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    assert(N == accumulate(begin(h_result), end(h_result), 0));

    for (auto i : h_result) {
        cout << i << "\n";
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}