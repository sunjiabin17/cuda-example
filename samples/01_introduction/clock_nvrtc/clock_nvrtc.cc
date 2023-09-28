// 使用NVIDIA Runtion Compilation (NVRTC) API编译CUDA代码
// 本例中，使用NVRTC编译一个简单的CUDA内核，然后在主机上运行它。

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <string.h>

#define NUM_BLOCKS 64
#define NUM_THREADS 256

#define KERNEL_FILE "/home/tars/projects/code/cuda_test/samples/01_introduction/clock_nvrtc/clock.cu"

#define NVRTC_SAFE_CALL(Name, x)                                \
  do {                                                          \
    nvrtcResult result = x;                                     \
    if (result != NVRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " << Name << " failed with error " \
                << nvrtcGetErrorString(result);                 \
      exit(1);                                                  \
    }                                                           \
  } while (0)



// 从文件中读取CUDA内核源代码并编译
void compileFileToCUBIN(const char* filename, char** cubin_result, size_t* cubin_size) {
    if (filename == nullptr) {
        throw std::runtime_error("readKernelCode: filename is nullptr");
        return ;
    }

    // FILE* fp = fopen(filename, "rb");   
    // if (fp == nullptr) {
    //     throw std::runtime_error("readKernelCode: open file failed");
    //     return ;
    // }
    std::ifstream ifs(filename, std::ios::in | std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw std::runtime_error("readKernelCode: open file failed");
        return ;
    }

    // tellg函数获取当前文件指针的位置，并将其存储在std::streampos类型的变量pos中
    std::streampos pos = ifs.tellg();
    size_t size = (size_t)pos;
    char* mem_block = new char[size + 1];
    // seekg函数将文件指针移动到文件开头。seekg函数的第一个参数是要移动的偏移量，第二个参数是移动的起始位置。
    ifs.seekg(0, std::ios::beg);
    ifs.read(mem_block, size);
    mem_block[size] = '\0';
    ifs.close();
    
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    std::cout << "GPU Compute Capability: " << major << "." << minor << std::endl;

    // 编译参数
    char* compile_params[2];
    std::string compile_option;
    compile_option = "-arch=sm_" + std::to_string(major) + std::to_string(minor);
    compile_params[0] = (char*)malloc(sizeof(char) * compile_option.size());
    strcpy(compile_params[0], compile_option.c_str());
    compile_params[1] = "--std=c++11";

    constexpr int num_compile_params = sizeof(compile_params) / sizeof(compile_params[0]);

    // 创建NVRTC实例
    nvrtcProgram prog;
    NVRTC_SAFE_CALL("nvrtcCreateProgram",
                    nvrtcCreateProgram(&prog,  // prog
                                       mem_block,  // buffer
                                       filename,  // name
                                       0,  // numHeaders
                                       NULL,  // headers
                                       NULL));  // includeNames
    NVRTC_SAFE_CALL("nvrtcCompileProgram",
                    nvrtcCompileProgram(prog,  // prog
                                        num_compile_params,  // numOptions
                                        compile_params));  // options
    size_t code_size;
    NVRTC_SAFE_CALL("nvrtcGetCUBINSize",
                    nvrtcGetCUBINSize(prog,  // prog
                                      &code_size));  // cubinSize
    char* cubin = new char[code_size];
    NVRTC_SAFE_CALL("nvrtcGetCUBIN",
                    nvrtcGetCUBIN(prog,  // prog
                                   cubin));  // cubin

    *cubin_result = cubin;
    *cubin_size = code_size;

    // for (int i = 0; i < num_compile_params; i++) {
    //     free(compile_params[i]);
    // }
}

CUmodule loadCUBIN(char* cubin, size_t cubin_size) {
    cuInit(0);
    CUcontext context;
    cuCtxCreate(&context, 0, 0);
    CUmodule module;
    // CUresult res = cuModuleLoadDataEx(&module, cubin, 0, 0, 0);
    CUresult res = cuModuleLoadData(&module, cubin);
    if (res != CUDA_SUCCESS) {
        free(cubin);
        throw std::runtime_error("loadCUBIN: cuModuleLoadDataEx failed");
    }
    free(cubin);
    return module;
}


int main(int argc, char** argv) {
    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    char* cubin, *kernel_file;
    size_t cubin_size;
    kernel_file = KERNEL_FILE;
    compileFileToCUBIN(kernel_file, &cubin, &cubin_size);
    CUmodule module = loadCUBIN(cubin, cubin_size);
    CUfunction kernel_addr;
    CUresult res = cuModuleGetFunction(&kernel_addr, module, "timeReduction");
    std::cout << res << std::endl;
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuModuleGetFunction failed");
    }

    dim3 grid(NUM_BLOCKS, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);
    CUdeviceptr d_input, d_output, d_time;
    cuMemAlloc(&d_input, sizeof(float) * NUM_THREADS * 2);
    cuMemAlloc(&d_output, sizeof(float) * NUM_BLOCKS);
    cuMemAlloc(&d_time, sizeof(clock_t) * NUM_BLOCKS * 2);
    cuMemcpyHtoD(d_input, input, sizeof(float) * NUM_THREADS * 2);

    void* args[] = {&d_input, &d_output, &d_time};
    cuLaunchKernel(kernel_addr, 
            grid.x, grid.y, grid.z,     // grid dim
            block.x, block.y, block.z,  // block dim
            sizeof(float) * NUM_THREADS * 2,  // shared mem
            0,  // stream
            args,  // args
            0);  // extra

    cuCtxSynchronize();
    cuMemcpyDtoH(timer, d_time, sizeof(clock_t) * NUM_BLOCKS * 2);
    cuMemFree(d_input);
    cuMemFree(d_output);
    cuMemFree(d_time);

    long double avg_elapsed_time = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        avg_elapsed_time += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }
    avg_elapsed_time /= NUM_BLOCKS;
    printf("Average elapsed time: %Lf\n", avg_elapsed_time);

    return 0;
}