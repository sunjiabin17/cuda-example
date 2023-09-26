all: build
build: clock.bin
PHONY: all build clean

# 查看显卡对应的计算能力
# nvidia-smi --query-gpu=name,compute_cap --format=csv
NVCC := nvcc
NVCCFLAGS := -std=c++11 -m64 -arch=sm_89 # --ptxas-options=-v -g -G

# @表示不显示命令本身
clock.o: clock.cu
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

clock.bin: clock.o
	@$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf *.o clock.bin