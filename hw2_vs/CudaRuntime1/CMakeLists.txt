cmake_minimum_required(VERSION 3.23)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(ee524_hw2
        LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 20)
include(FindCUDAToolkit)

add_library(kernels kernels/include/addKernel.cuh kernels/source/addKernel.cu)
set_target_properties(kernels
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(kernels PUBLIC
        kernels/include)


add_executable(main
        #apps/main.cpp
        apps/hw2_starter.cu
        )
set_source_files_properties(apps/main.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(main
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main
        PUBLIC
        )
target_link_libraries(main
        PUBLIC
        kernels
        CUDA::cudart
        )
