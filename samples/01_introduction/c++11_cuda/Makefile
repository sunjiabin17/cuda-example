all: build
build: c++11_cuda
PHONY: all build clean

# 查看显卡对应的计算能力
# nvidia-smi --query-gpu=name,compute_cap --format=csv
NVCC := nvcc
NVCCFLAGS := -std=c++11 -m64 -arch=sm_89 -g -G

# @表示不显示命令本身
c++11_cuda.o: c++11_cuda.cu
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

c++11_cuda: c++11_cuda.o
	@$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf *.o c++11_cuda