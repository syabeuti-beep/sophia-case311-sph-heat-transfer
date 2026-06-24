make :
	nvcc -w -use_fast_math -arch=sm_60 -O3 -expt-relaxed-constexpr SOPHIA_gpu.cu -o SOPHIA_gpu -I./cub-1.8.0/ -lpthread
clean :
	rm -rf *.*~ SOPHIA_gpu
