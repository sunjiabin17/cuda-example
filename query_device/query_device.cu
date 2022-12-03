#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void query_device() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found");
    }
    int devNo = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devNo);
    printf("Device %d: \"%s\"\n", devNo, deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %d KB\n", (deviceProp.totalGlobalMem >> 10));
    printf("  Total constant memory: %d KB\n", (deviceProp.totalConstMem >> 10));
    printf("  Total shared memory per block: %d KB\n", (deviceProp.sharedMemPerBlock >> 10));
    printf("  TOtal shared memory per multiprocessor: %d KB\n", (deviceProp.sharedMemPerMultiprocessor >> 10));
    printf("  Total registers per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size: %d\n", deviceProp.warpSize);
    printf("  Maximum memory pitch: %d\n", deviceProp.memPitch);
    printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Maximum threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
    for (int i = 0; i < 3; ++i)
        printf("  Maximum dimension %d of block: %d\n", i, deviceProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("  Maximum dimension %d of grid: %d\n", i, deviceProp.maxGridSize[i]);
    printf("  Clock rate: %d\n", deviceProp.clockRate);
    printf("  Texture alignment: %d\n", deviceProp.textureAlignment);
    printf("  Concurrent copy and execution: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
    printf("  Kernel execution timeout : %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");

}
int main() {
    query_device();

    return 0;
}