# 指定CMake的最小版本要求
cmake_minimum_required(VERSION 3.0)

# 定义工程名称
project(CUDA_TEST LANGUAGES CXX CUDA)

enable_language(CUDA)


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)




# find_package(CUDA REQUIRED)
# set(CUDA_TOOLKIT_DIR "/usr/local/cuda")

# set(SRC kernel.cu)
file(GLOB SRC *.cu)

# 向工程添加多个特定的头文件搜索路径
# include_directories(
#     ${CMAKE_SOURCE_DIR}/include
# )

# 可执行文件输出的存放路径
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# set(CMAKE_BUILD_TYPE Debug)
# set(SKIP_BUILD_RPATH ON)

# 生成可执行文件
add_executable(cuda_bin ${SRC})

# target_link_libraries(cuda_bin ${CUDA_LIBRARIES})
# target_compile_options(cuda_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                        --generate-line-info
#                        --use_fast_math
#                        --relocatable-device-code=true
#                        -lcudadevrt
#                        -arch=sm_61
#                        >)