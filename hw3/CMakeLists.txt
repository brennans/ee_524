cmake_minimum_required(VERSION 3.23)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(ee524_hw3
        LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 20)
include(FindCUDAToolkit)


add_executable(main
            CudaRuntime1/CudaRuntime1/kernel.cu
        )
set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(main
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main
        PUBLIC
        include
        )
target_link_libraries(main
        PUBLIC
        CUDA::cudart
        )
