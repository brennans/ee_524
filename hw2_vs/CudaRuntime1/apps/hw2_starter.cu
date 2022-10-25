
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "windows.h"
#include "profileapi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <chrono>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION // this is needed
#include "../util/stb_image.h"  // download from class website files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image_write.h"  // download from class website files

// #include your error-check macro header file here
#include "../include/check_cuda.h"

// global gaussian blur filter coefficients array here
#define BLUR_FILTER_WIDTH 9  // 9x9 (square) Gaussian blur filter
const float BLUR_FILT[81] = { 0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084,
                              0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,
                              0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,
                              0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,
                              0.3292,0.5353,0.7575,0.9329,1.0000,0.9329,0.7575,0.5353,0.3292,
                              0.3071,0.4994,0.7066,0.8703,0.9329,0.8703,0.7066,0.4994,0.3071,
                              0.2494,0.4054,0.5738,0.7066,0.7575,0.7066,0.5738,0.4054,0.2494,
                              0.1762,0.2865,0.4054,0.4994,0.5353,0.4994,0.4054,0.2865,0.1762,
                              0.1084,0.1762,0.2494,0.3071,0.3292,0.3071,0.2494,0.1762,0.1084};
#define BLUR_FILTER_HALF_WIDTH (BLUR_FILTER_WIDTH / 2)


// DEFINE your CUDA blur kernel function(s) here
// blur kernel #1 - global memory only
__global__ void blurKernelGlobal(uint8_t *out, const uint8_t *in, const float *blur, const int width, const int height)
{
    int column_index = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row_index * width + column_index;

    if (column_index < width && row_index < height) {
        int pixel_value = 0;
        float normalization = 0;

        for (int blurRow = -1 * (BLUR_FILTER_HALF_WIDTH) + row_index, blurKernelRowIndex = 0; blurKernelRowIndex < BLUR_FILTER_WIDTH; blurRow++, blurKernelRowIndex++) {
            for (int blurColumn = -1 * (BLUR_FILTER_HALF_WIDTH) + column_index, blurKernelColIndex = 0; blurKernelColIndex < BLUR_FILTER_WIDTH; blurColumn++, blurKernelColIndex++) {
                if (blurRow > 0 && blurRow < width && blurColumn > 0 && blurColumn < height)  {
                    float blur_coeff = blur[blurKernelRowIndex * BLUR_FILTER_WIDTH  + blurKernelColIndex];
                    pixel_value += round(in[blurRow * width + blurColumn] * blur_coeff);
                    normalization += blur_coeff;
                }
            }
        }
        out[index] = static_cast<uint8_t>(pixel_value / normalization);
    }

}


// blur kernel #2 - device shared memory (static alloc)
__global__ void blurKernelStatic(uint8_t *out, const uint8_t *in, const float* blur_in, const int width, const int height)
{

    __shared__ float blur[BLUR_FILTER_WIDTH][BLUR_FILTER_WIDTH];

    int column_index = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row_index * width + column_index;
    if (threadIdx.x < BLUR_FILTER_WIDTH  && threadIdx.y < BLUR_FILTER_WIDTH) {
        blur[threadIdx.y][threadIdx.x] = blur_in[BLUR_FILTER_WIDTH * threadIdx.y + threadIdx.y];
    }
    __syncthreads();

    if (column_index < width && row_index < height) {
        int pixel_value = 0;
        float normalization = 0;

        for (int blurRow = -1 * (BLUR_FILTER_HALF_WIDTH) + row_index, blurKernelRowIndex = 0; blurKernelRowIndex < BLUR_FILTER_WIDTH; blurRow++, blurKernelRowIndex++) {
            for (int blurColumn = -1 * (BLUR_FILTER_HALF_WIDTH) + column_index, blurKernelColIndex = 0; blurKernelColIndex < BLUR_FILTER_WIDTH; blurColumn++, blurKernelColIndex++) {
                if (blurRow > 0 && blurRow < width && blurColumn > 0 && blurColumn < height)  {
                    float blur_coeff = blur[blurKernelRowIndex][blurKernelColIndex];
                    pixel_value += static_cast<int>(round(in[blurRow * width + blurColumn] * blur_coeff));
                    normalization += blur_coeff;
                }
            }
        }
        out[index] = static_cast<uint8_t>(pixel_value / normalization);
    }

}


// blur kernel #3 - device shared memory (dynamic alloc)
__global__ void blurKernelDynamic(uint8_t *out, const uint8_t *in, const float* blur_in, const int width, const int height)
{
    extern __shared__ float blur[];
    int column_index = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row_index * width + column_index;

    if (threadIdx.x < BLUR_FILTER_WIDTH  && threadIdx.y < BLUR_FILTER_WIDTH) {
        blur[BLUR_FILTER_WIDTH * threadIdx.y + threadIdx.y] = blur_in[BLUR_FILTER_WIDTH * threadIdx.y + threadIdx.y];
    }
    __syncthreads();

    if (column_index < width && row_index < height) {
        int pixel_value = 0;
        float normalization = 0;

        for (int blurRow = -1 * (BLUR_FILTER_HALF_WIDTH) + row_index, blurKernelRowIndex = 0; blurKernelRowIndex < BLUR_FILTER_WIDTH; blurRow++, blurKernelRowIndex++) {
            for (int blurColumn = -1 * (BLUR_FILTER_HALF_WIDTH) + column_index, blurKernelColIndex = 0; blurKernelColIndex < BLUR_FILTER_WIDTH; blurColumn++, blurKernelColIndex++) {
                if (blurRow > 0 && blurRow < width && blurColumn > 0 && blurColumn < height)  {
                    float blur_coeff = blur[blurKernelRowIndex * BLUR_FILTER_WIDTH  + blurKernelColIndex];
                    pixel_value += round(in[blurRow * width + blurColumn] * blur_coeff);
                    normalization += blur_coeff;
                }
            }
        }
        out[index] = static_cast<uint8_t>(pixel_value / normalization);
    }

}

// EXTRA CREDIT
// define host sequential blur-kernel routine

enum class KernelType {
    GLOBAL = 0,
    STATIC_MEM = 1,
    DYNAMIC_MEM = 2,
};


uint64_t host_blur(uint8_t *out, const uint8_t *in, const float *blur, const int width, const int height) {
    auto host_compute_start  = std::chrono::steady_clock::now();

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int index = width * i + j;
            int pixel_value = 0;
            float normalization = 0;

            for (int blurRow = -1 * (BLUR_FILTER_HALF_WIDTH) + i, blurKernelRowIndex = 0; blurKernelRowIndex < BLUR_FILTER_WIDTH; blurRow++, blurKernelRowIndex++) {
                for (int blurColumn = -1 * (BLUR_FILTER_HALF_WIDTH) + j, blurKernelColIndex = 0; blurKernelColIndex < BLUR_FILTER_WIDTH; blurColumn++, blurKernelColIndex++) {
                    if (blurRow > 0 && blurRow < width && blurColumn > 0 && blurColumn < height)  {
                        float blur_coeff = blur[blurKernelRowIndex * BLUR_FILTER_WIDTH  + blurKernelColIndex];
                        pixel_value += round(in[blurRow * width + blurColumn] * blur_coeff);
                        normalization += blur_coeff;
                    }
                }
            }
            out[index] = static_cast<uint8_t>(pixel_value / normalization);
        }
    }


    auto host_compute_time  = std::chrono::steady_clock::now() - host_compute_start;

    return host_compute_time.count();

}

int runKernel(const char* in_filename, const char* out_filename, KernelType kernelType, std::vector<uint64_t> &time_1, std::vector<uint64_t> &time_2) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }


    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char *imgData = stbi_load(in_filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * sizeof(uint8_t);

    // setup additional host variables, allocate host memory as needed
    char *h_imgOut = (char*)malloc(imgSize);
    if (h_imgOut == NULL) {
        printf("Error! Out of host memory!");
        return -1;
    }
    std::cout << "  Processing " << in_filename << std::endl;
    std::cout << "  Image Parameters. Size: " << imgSize << ", Width: " << x_cols << ", Height: " << y_rows << "\n";
    dim3 numThreads {32, 32, 1};
    dim3 numBlocks {static_cast<unsigned int>(x_cols / 32) + 1, static_cast<unsigned int>(y_rows / 32) + 1, 1};
    std::cout << "  Using Kernel Parameters... \n"
                 "    Block Dimensions: (" <<  numBlocks.x << ", " <<  numBlocks.y << ") ";
    std::cout << "    Thread Dimensions: (" << numThreads.x << ", " << numThreads.y << ")" << std::endl;


    // START timer #1
    auto timer_1_start  = std::chrono::steady_clock::now();
    LARGE_INTEGER timer_1_start_windows;
    QueryPerformanceCounter(&timer_1_start_windows);

    // allocate device memory
    uint8_t *dev_in_image;
    uint8_t *dev_out_image;
    float *dev_blur_global;
    checkCuda(cudaMalloc((void**)&dev_in_image, imgSize * sizeof(uint8_t)));
    checkCuda(cudaMalloc((void**)&dev_out_image, imgSize * sizeof(uint8_t)));
    checkCuda(cudaMalloc((void**)&dev_blur_global, sizeof(BLUR_FILT)));

    // copy host data to device
    checkCuda(cudaMemcpy(dev_in_image, imgData, imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_blur_global, BLUR_FILT, sizeof(BLUR_FILT), cudaMemcpyHostToDevice));


    // START timer #2
    auto timer_2_start = std::chrono::steady_clock::now();
    LARGE_INTEGER timer_2_start_windows;
    QueryPerformanceCounter(&timer_2_start_windows);

    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range
    if (kernelType == KernelType::GLOBAL) {
        blurKernelGlobal<<<numBlocks, numThreads>>>(dev_out_image, dev_in_image, dev_blur_global, x_cols, y_rows);
    } else if(kernelType == KernelType::STATIC_MEM) {
        blurKernelStatic<<<numBlocks, numThreads>>>(dev_out_image, dev_in_image, dev_blur_global, x_cols, y_rows);
    } else if(kernelType == KernelType::DYNAMIC_MEM) {
        blurKernelGlobal<<<numBlocks, numThreads, sizeof(BLUR_FILT)>>>(dev_out_image, dev_in_image, dev_blur_global, x_cols, y_rows);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "blurKernelX launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());


    // STOP timer #2
    //
    // retrieve result data from device back to host
    auto timer_2 = std::chrono::steady_clock::now() - timer_2_start;
    LARGE_INTEGER timer_2_end_windows;
    QueryPerformanceCounter(&timer_2_end_windows);
    int64_t timer_2_windows = timer_2_end_windows.QuadPart - timer_2_start_windows.QuadPart;

    checkCuda(cudaMemcpy(h_imgOut, dev_out_image, imgSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // STOP timer #1
    auto timer_1 = std::chrono::steady_clock::now() - timer_1_start;
    LARGE_INTEGER timer_1_end_windows;
    QueryPerformanceCounter(&timer_1_end_windows);
    int64_t timer_1_windows = timer_1_end_windows.QuadPart - timer_1_start_windows.QuadPart;

    // save result output image data to file
    stbi_write_png(out_filename, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    std::cout << "  Total computation time value: " << timer_1.count() << "ns, Windows: " << (timer_1_windows * 1000000) / freq.QuadPart << " us" << std::endl;
    std::cout << "  GPU kernel time: " << timer_2.count() << "ns, Windows: " << (timer_2_windows * 1000000) / freq.QuadPart << " us" << std::endl;
    time_1.push_back(timer_1.count());
    time_2.push_back(timer_2.count());


    // EXTRA CREDIT:
    // start timer #3
    // run host sequential blur routine
    // stop timer #3

    // retrieve and save timer results (write to console or file)

    Error:  // assumes error macro has a goto Error statement

    // free host and device memory
    cudaFree(dev_in_image);
    cudaFree(dev_out_image);
    cudaFree(dev_blur_global);
    free(h_imgOut);
    return cudaStatus != cudaSuccess ? -1 : 0;
}


int main() {

    // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char* in_filenames[] {
            "../data/hw2_testimage1.png",
            "../data/hw2_testimage2.png",
            "../data/hw2_testimage3.png",
            "../data/hw2_testimage4.png"
    };
    const char* out_filenames[] {
            "../hw2_outimage1.png",
            "../hw2_outimage2.png",
            "../hw2_outimage3.png",
            "../hw2_outimage4.png"
    };
    const char filename[] = "../data/hw2_testimage3.png";
    std::vector<uint64_t> global_timer_1;
    std::vector<uint64_t> global_timer_2;
    for (int i = 0; i < 3; i++) {
        std::cout << "Running Global Kernel" << std::endl;
        int ret = runKernel(in_filenames[i], out_filenames[i], KernelType::GLOBAL, global_timer_1, global_timer_2);
        if (ret != 0) {
            printf("Error running global kernel. \n");
            return -1;
        }
    }
    std::vector<uint64_t> static_timer_1;
    std::vector<uint64_t> static_timer_2;
    for (int i = 0; i < 3; i++) {
        std::cout << "Running Static Memory Allocation Kernel" << std::endl;
        int ret = runKernel(in_filenames[i], out_filenames[i], KernelType::STATIC_MEM, static_timer_1, static_timer_2);
        if (ret != 0) {
            printf("Error running Static Memory Allocation kernel. \n");
            return -1;
        }
    }
    std::vector<uint64_t> dyn_timer_1;
    std::vector<uint64_t> dyn_timer_2;

    for (int i = 0; i < 3; i++) {
        std::cout << "Running Dynamic Memory Allocation Kernel" << std::endl;
        int ret = runKernel(in_filenames[i], out_filenames[i], KernelType::DYNAMIC_MEM, dyn_timer_1, dyn_timer_2);
        if (ret != 0) {
            printf("Error running Dynamic Memory Allocation kernel. \n");
            return -1;
        }
    }

    std::vector<uint64_t> host_timer;
    for (int i = 0; i < 3; i++) {
        std::cout << "Computing using host function" << std::endl;
        int x_cols = 0;
        int y_rows = 0;
        int n_pixdepth = 0;
        unsigned char *imgData = stbi_load(in_filenames[i], &x_cols, &y_rows, &n_pixdepth, 1);
        int imgSize = x_cols * y_rows * sizeof(uint8_t);

        // setup additional host variables, allocate host memory as needed
        char *h_imgOut = (char*)malloc(imgSize);
        if (h_imgOut == NULL) {
            printf("Error! Out of host memory!");
            return -1;
        }

        uint64_t host_time = host_blur((uint8_t*)h_imgOut, (uint8_t*)imgData, &BLUR_FILT[0], x_cols, y_rows);

        host_timer.push_back(host_time);
    }

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());

    printf("%20s  %14s, %14s, %14s, %14s, %14s, %14s, %14s\n", "(Nanoseconds)", "Global Timer 1", "Global Timer 1", "Static Timer 1","Static Timer 2",  "Dynamic Timer 1", "Dynamic Timer 2", "Host Timer");
    for (int i = 0; i < global_timer_1.size(); i++) {
        printf("%10s: %14llu, %14llu, %14llu, %14llu, %14llu, %14llu, %14llu\n", out_filenames[i],
               global_timer_1[i], global_timer_2[i], static_timer_1[i], static_timer_2[i], dyn_timer_1[i], dyn_timer_2[i], host_timer[i]);
    }

    return 0;
}


