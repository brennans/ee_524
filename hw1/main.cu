
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <array>
#include <vector>
#include <cstdint>

#include <stdio.h>
#include <iostream>
#include <cassert>

#include "check_cuda.h"
#include "matrix_2d_add.cuh"
#include "matrix_3d_add.cuh"


bool equals(double x, double y, double absTol, double relTol){
    if (std::abs(x - y) <= std::max(absTol, relTol * std::max(std::abs(x), std::abs(y)))) {
        return true;
    } else {
        return false;
    }
}

bool compare_floats(float x, float y, float absTol){
    if (std::abs(x - y) <= absTol) {
        return true;
    } else {
        printf("Check failed: %f, %f\n", x, y);
        return false;
    }
}



// Allocate memory on the GPU given a set of host memory
// Adds
template <class T>
cudaError_t allocateDeviceBuffers(std::vector<std::vector<T>> &hostVecs, std::vector<T*> deviceVecs){
    for (int i = 0; i < hostVecs.size(); i++) {
        T* deviceBuf = nullptr;
        auto&& hostVec = hostVecs[i];
        cudaError_t cudaStatus = cudaMalloc((void**)&deviceBuf, hostVec.size() * sizeof(T));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }


        deviceVecs.push_back(deviceBuf);
    }
}

__global__ void printKernel(int *c, const int *a, const int *b)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    printf("    Block Id: %d, Thread Id: %d, Global Index: %d\n", blockIdx.x, threadIdx.x, i);
}


__global__ void saxpyKernel(float *z, const float *x, const float *y, const float a, const int N)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        z[i] = a * x[i] + y[i];
    }
}

void checkSaxpyKernel(const float *z, const float*x, const float *y, const float a, const int N){
    for (int i = 0; i < N; i++) {
        float result = a * x[i] + y[i];
        assert(compare_floats(z[i], result, 10));
    }
}


// Helper function for using CUDA
cudaError_t runPrintfKernel(int *c, const int *a, const int *b, uint32_t numBlocks, uint32_t numThreads, int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    printKernel<<<numBlocks, numThreads>>>(dev_c, dev_a, dev_b);


    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


// Helper function for using CUDA
cudaError_t runSaxpyKernel(float *c, const float *a, const float *b, uint32_t numBlocks, uint32_t numThreads, int size)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    saxpyKernel<<<numBlocks, numThreads>>>(dev_c, dev_a, dev_b, 10, size);


    // Check for any errors launching the kernel
    checkCuda(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



template<int N>
static constexpr std::array<int, N> createArray(int mult)
{
    std::array<int, N> a {};
    for (int i = 0; i < a.size(); i++) {
        a[i] = i * mult;
    }
    return a;
}


int main()
{

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }



    // Run printfKernel
    for (int i = 0; i < 10; i++) {
        printf("Run %d\n", i);
        const int arraySize = 16;
        std::array<int, arraySize> a = createArray<arraySize>(1);
        std::array<int, arraySize> b = createArray<arraySize>(10);
        int c[arraySize] = { 0 };

        const uint32_t numBlocks = 4;
        const uint32_t numThreads = 4;
        static_assert(numBlocks * numThreads == arraySize, "Grid and Block dimensions must match the array size");

        cudaStatus = checkCuda(runPrintfKernel(c, a.data(), b.data(), numBlocks, numThreads, arraySize));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "runCudaKernel failed!");
            return 1;
        }
    }

    // Run the SAXPY kernel
    const std::array<int, 3> test_sizes = {10, static_cast<int>(1e3), static_cast<int>(1e7)};

    for (int i = 0; i < test_sizes.size(); i++) {
        printf("Run %d\n", i);
        int size = test_sizes[i];

        // Create vectors with test case sizes
        std::vector<float> x(size);
        std::vector<float> y(size);
        for (int j = 0; j < size; j++) {
            x[j] = j;
            y[j] = j;
        }

        std::vector<float> z(size, 0);
        uint32_t numThreads = 1024;
        uint32_t numBlocks = (size + numThreads - 1) / numThreads;

        cudaStatus = runSaxpyKernel( z.data(), x.data(), y.data(), numBlocks, numThreads, size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "runCudaKernel failed for saxypy!");
            return 1;
        }

        for (int j = 0; j < 10; j++) {
            std::cout << z[j] << ", ";
        }
        if (size > 10) {
            std::cout << "...";
        }
        std::cout << std::endl;

        checkSaxpyKernel(z.data(), x.data(), y.data(), 10, size);
    }


    // Run the 2D Matrix Add
    std::array<dim3, 3> test_sizes_2d {{
            {{4, 4, 1}},
            {{500, 500, 1}},
            {{3024, 4032, 1}}
    }};
    for (int i = 0; i < test_sizes.size(); i++) {
        printf("Run %d\n", i);

        dim3 currentDim = test_sizes_2d[i];
        uint32_t size = currentDim.x * currentDim.y;

        // Create vectors with test case sizes
        std::vector<int> a(size);
        std::vector<int> b(size);
        for (int j = 0; j < size; j++) {
            a[j] = j + 0.1;
            b[j] = j;
        }
        std::vector<int> c(size, 0);

        uint32_t threadDim = std::min(currentDim.x, 32u);

        cudaStatus = runMatrix2dAdd(c.data(), a.data(), b.data(), currentDim.y, currentDim.x, threadDim);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "runCudaKernel failed for runMatrix2dAdd!");
            return 1;
        }

        checkMatrix2dAdd(c.data(), a.data(), b.data(), currentDim.y, currentDim.x);

    }

    // Run the 3D Grid Add
    std::array<dim3, 3> test_sizes_3d {{
                                               {{3, 3, 3}},
                                               {{100, 100, 100}},
                                               {{1000, 1000, 1000}}
                                       }};
    for (int i = 0; i < test_sizes.size(); i++) {
        printf("Run %d\n", i);

        dim3 currentDim = test_sizes_2d[i];
        uint32_t size = currentDim.x * currentDim.y;

        // Create vectors with test case sizes
        std::vector<float> a(size);
        std::vector<float> b(size);
        for (int j = 0; j < size; j++) {
            a[j] = static_cast<float>(j) + 0.1;
            b[j] = j + 1000000;
        }
        std::vector<float> c(size, 0);

        uint32_t threadDim = std::min(currentDim.x, 8u);

        cudaStatus = runMatrix3dAdd(c.data(), a.data(), b.data(), currentDim.y, currentDim.x, currentDim.z, threadDim);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "runCudaKernel failed for runMatrix2dAdd!");
            return 1;
        }

        checkMatrix3dAdd(c.data(), a.data(), b.data(), currentDim.y, currentDim.x, currentDim.z);

    }



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}





