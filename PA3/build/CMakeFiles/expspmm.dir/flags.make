# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# compile CUDA with /usr/local/cuda-11.1/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = 

CUDA_INCLUDES = -I/home/course/hpc/users/2020010944/PA3/./include -I/home/course/hpc/users/2020010944/local/include -I/home/course/hpc/users/2020010944/PA3/third_party

CUDA_FLAGS = -Xcompiler -fopenmp   -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -O3 --use_fast_math -lcurand   -std=c++14

CXX_DEFINES = 

CXX_INCLUDES = -I/home/course/hpc/users/2020010944/PA3/./include -I/home/course/hpc/users/2020010944/local/include -I/home/course/hpc/users/2020010944/PA3/third_party

CXX_FLAGS =  -fopenmp

