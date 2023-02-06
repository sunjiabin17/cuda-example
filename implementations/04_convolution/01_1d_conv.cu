#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void convolution_1d(int *input, int *mask, int *output, int n,
                               int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = m / 2;
    int start = tid - r;
    int temp = 0;

    for (int i = 0; i < m; ++i) {
        if (start + i >= 0 && start + i < n) {
            temp += input[start + i] * mask[i];
        }
    }
    output[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *input, int *mask, int *output, int n, int m) {
    int radius = m / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < m; j++) {
            if ((start + j >= 0) && (start + j < n)) {
            temp += input[start + j] * mask[j];
            }
        }
        assert(temp == output[i]);
    }
}

int main() {
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    int m = 7;
    int bytes_m = m * sizeof(int);

    std::vector<int> h_input(n);

    std::generate(begin(h_input), end(h_input), [](){ return rand() % 100; });

    std::vector<int> h_mask(m);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

    std::vector<int> h_output(n, 0);

    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_array, h_input.data(), bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

    int THREADS = 256;

    int GRID = (n + THREADS - 1) / THREADS;

    convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

    cudaMemcpy(h_output.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_input.data(), h_mask.data(), h_output.data(), n, m);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_array);

    return 0;
}