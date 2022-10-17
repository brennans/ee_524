cmake_minimum_required(VERSION 3.23)
project(hw1 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(hw1 main.cu
        include/check_cuda.h
        include/matrix_2d_add.cuh
        include/matrix_3d_add.cuh)
target_include_directories(hw1
        PUBLIC
        include
        )

set_target_properties(hw1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
