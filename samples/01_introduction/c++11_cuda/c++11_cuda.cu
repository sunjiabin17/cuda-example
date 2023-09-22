#include <iostream>

template<typename T, typename Predicate>
__device__ void count_if(int* count, T* data, int n, Predicate p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (p(data[tid])) {
            atomicAdd(count, 1);
        }
    }
}

__global__ void xyzw_frequency(int* count, char* text, int len) {
    const char letters[] = {'x', 'y', 'z', 'w'};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    count_if(count, text, len, [&](char c) {
        for (int i = 0; i < 4; i++) {
            if (c == letters[i]) {
                return true;
            }
        }
        return false;
    });
}

int main(int argc, char** argv) {
    constexpr char* filename = "warandpeace.txt";
    int numBytes = 16 * 1048576;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    char* h_text = (char*)malloc(numBytes);
    char* d_text;
    cudaMalloc(&d_text, numBytes);

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        std::cout << "Error opening file" << std::endl;
        return 1;
    }
    int len = (int)fread(h_text, sizeof(char), numBytes, file);
    fclose(file);
    std::cout << "read bytes: " << len << std::endl;
    cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice);

    int count = 0;
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    xyzw_frequency<<<blocksPerGrid, threadsPerBlock>>>(d_count, d_text, len);
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "elapsed time: " << elapsedTime << " ms" << std::endl;
    std::cout << "count: " << count << std::endl;

    cudaFree(d_text);
    cudaFree(d_count);
    free(h_text);
    return 0;
}