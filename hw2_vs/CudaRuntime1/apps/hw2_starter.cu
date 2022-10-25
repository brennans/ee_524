
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <chrono>

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
        float pixel_value = 0;
        int number_of_pixels = 0;

        for (int blurRow = -1 * (BLUR_FILTER_HALF_WIDTH) + row_index, blurKernelRowIndex = 0; blurKernelRowIndex < BLUR_FILTER_WIDTH; blurRow++, blurKernelRowIndex++) {
            for (int blurColumn = -1 * (BLUR_FILTER_HALF_WIDTH) + column_index, blurKernelColIndex = 0; blurKernelColIndex < BLUR_FILTER_WIDTH; blurColumn++, blurKernelColIndex++) {
                if (blurRow > 0 && blurRow < width && blurColumn > 0 && blurColumn < height)  {
                    pixel_value += in[blurRow * width + blurColumn] * blur[blurKernelRowIndex * BLUR_FILTER_WIDTH  + blurKernelColIndex];
                    number_of_pixels += 1;
                }
            }
        }
        out[index] = static_cast<uint8_t>(pixel_value / number_of_pixels);
    }

}


// blur kernel #2 - device shared memory (static alloc)
__global__ void blurKernelStatic(uint8_t *out, const uint8_t *in, const float *blur, const int width, const int height)
{
    int column_index = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row_index * width + column_index;

    if (column_index < width && row_index < height) {
        float pixel_value = 0;
        int number_of_pixels = 0;

        for (int blurRow = -1 * (BLUR_FILTER_HALF_WIDTH) + row_index, blurKernelRowIndex = 0; blurKernelRowIndex < BLUR_FILTER_WIDTH; blurRow++, blurKernelRowIndex++) {
            for (int blurColumn = -1 * (BLUR_FILTER_HALF_WIDTH) + column_index, blurKernelColIndex = 0; blurKernelColIndex < BLUR_FILTER_WIDTH; blurColumn++, blurKernelColIndex++) {
                if (blurRow > 0 && blurRow < width && blurColumn > 0 && blurColumn < height)  {
                    pixel_value += in[blurRow * width + blurColumn] * blur[blurKernelRowIndex * BLUR_FILTER_WIDTH  + blurKernelColIndex];
                    number_of_pixels += 1;
                }
            }
        }
        out[index] = static_cast<uint8_t>(pixel_value / number_of_pixels);
    }

}


// blur kernel #2 - device shared memory (dynamic alloc)


// EXTRA CREDIT
// define host sequential blur-kernel routine



int main() {

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // read input image from file - be aware of image pixel bit-depth and resolution (horiz x vertical)
    const char filename[] = "../data/hw2_testimage1.png";
    int x_cols = 0;
    int y_rows = 0;
    int n_pixdepth = 0;
    unsigned char *imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
    int imgSize = x_cols * y_rows * sizeof(uint8_t);

    // setup additional host variables, allocate host memory as needed
    char *h_imgOut = (char*)malloc(imgSize);
    if (h_imgOut == NULL) {
        printf("Out of host memory!");
        return -1;
    }
    std::cout << "Image Parameters. Size: " << imgSize << ", Width: " << x_cols << ", Height: " << y_rows << "\n";
    dim3 numThreads {32, 32, 1};
    dim3 numBlocks {static_cast<unsigned int>(x_cols / 32) + 1, static_cast<unsigned int>(y_rows / 32) + 1, 1};
    std::cout << "Using Kernel Parameters... \n  Block Dimensions: (" <<  numBlocks.x << ", " <<  numBlocks.y << ") ";
    std::cout << "  Thread Dimensions: (" << numThreads.x << ", " << numThreads.y << ")" << std::endl;


    // START timer #1
    auto timer_1_start  = std::chrono::steady_clock::now();

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

    // launch kernel --- use appropriate heuristics to determine #threads/block and #blocks/grid to ensure coverage of your 2D data range
    blurKernelGlobal<<<numBlocks, numThreads>>>(dev_out_image, dev_in_image, dev_blur_global, x_cols, y_rows);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // call cudaDeviceSynchronize() to wait for the kernel to finish, and return
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());


    // STOP timer #2
    // 
    // retrieve result data from device back to host
    auto timer_2 = std::chrono::steady_clock::now() - timer_2_start;
    checkCuda(cudaMemcpy(h_imgOut, dev_out_image, imgSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // STOP timer #1
    auto timer_1 = std::chrono::steady_clock::now() - timer_1_start;

    // cudaDeviceReset( ) must be called in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset());


       // save result output image data to file
    const char imgFileOut[] = "./hw2_outimage1.png";
    stbi_write_png(imgFileOut, x_cols, y_rows, 1, h_imgOut, x_cols * n_pixdepth);

    std::cout << "Total computation time value: " << timer_1.count() << std::endl;
    std::cout << "GPU kernel time: " << timer_2.count() << std::endl;


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


    return 0;
}


