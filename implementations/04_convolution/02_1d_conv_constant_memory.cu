#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CONV_1D_KERNEL 7
__constant__ int d_mask[CONV_1D_KERNEL];

__global__ void convolution_1d(int *input, int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int r = CONV_1D_KERNEL / 2;
    int start = tid - r;
    int temp = 0;

    for (int i = 0; i < CONV_1D_KERNEL; ++i) {
        if (start + i >= 0 && start + i < n) {
            temp += input[start + i] * d_mask[i];
        }
    }
    output[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *input, int *mask, int *output, int n) {
    int radius = CONV_1D_KERNEL / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < CONV_1D_KERNEL; j++) {
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
    int bytes_m = CONV_1D_KERNEL * sizeof(int);

    std::vector<int> h_input(n);

    std::generate(begin(h_input), end(h_input), [](){ return rand() % 100; });

    std::vector<int> h_mask(CONV_1D_KERNEL);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

    std::vector<int> h_output(n, 0);

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes_n);
    cudaMalloc(&d_output, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_input, h_input.data(), bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask.data(), bytes_m);

    int THREADS = 256;

    int GRID = (n + THREADS - 1) / THREADS;

    convolution_1d<<<GRID, THREADS>>>(d_input, d_output, n);

    cudaMemcpy(h_output.data(), d_output, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_input.data(), h_mask.data(), h_output.data(), n);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_output);
    cudaFree(d_input);

    return 0;
}