#ifndef HW1_CHECK_CUDA_H
#define HW1_CHECK_CUDA_H

#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

#endif //HW1_CHECK_CUDA_H
