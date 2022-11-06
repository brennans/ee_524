#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <check_cuda.h>


__global__ void singleBifurcationKernel(int *b, const int *a, const int index) {
    int i = threadIdx.x;
    if (i < index) {
        b[i] = a[i];
    } else {
        b[i] = -1 * a[i];
    }
}

__global__ void nBifurcationKernel(int *b, const int *a, int min) {
    int n = threadIdx.x;
    if (n < min) {
        n = min;
    }
    for (int i = 0; i < n; i++) {
        b[threadIdx.x] = a[i] + a[i];
    }
}

enum class BifurcationType {
    Single,
    N
};


// Helper function for using CUDA to add vectors in parallel.
cudaError_t runBifucationKernel(BifurcationType type, unsigned int size) {

    int *a = (int *) malloc(size * sizeof(int));
    int *b = (int *) malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        a[i] = i;
    }

    int *dev_a = 0;
    int *dev_b = 0;
    cudaError_t cudaStatus;


    // Allocate GPU buffers for three vectors (one input, one output)
    checkCuda(cudaMalloc((void **) &dev_a, size * sizeof(int)));
    checkCuda(cudaMalloc((void **) &dev_b, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    if (BifurcationType::Single == type) {
        singleBifurcationKernel<<<4, size>>>(dev_b, dev_a, size/2);
    } else if (BifurcationType::N == type) {
        nBifurcationKernel<<<4, size>>>(dev_b, dev_a, 0);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++) {
        printf(" %d, ", b[i]);
        if ((i + 1) % 32 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    Error:
    cudaFree(dev_a);
    cudaFree(dev_b);

    free(a);
    free(b);

    return cudaStatus;
}


/*
 * Takes two matrices M (l x m) and N (m x n) and computes the product into the matrix P (l x n).
 */
void matrix_multiplication(const float *M, const float *N, float *P, const int l, const int m, const int n) {

    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            int out_index = i * n + j;
            P[out_index] = 0;
            for (int k = 0; k < m; k++) {
                int m_index = m * i + k;
                int n_index = n * k + j;
                float val = M[m_index] * N[n_index];
                P[out_index] += val;
            }
        }
    }
}

__global__ void mmultiKernel(float *P, const float *M, const float *N, const int l, const int m, const int n) {
    // Compute output matrix row and column
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * n + column;

    // Check we have valid indicies in the matrix
    if (row < l && column < n) {
        P[index] = 0;
        for (int i = 0; i < m; i++) {
            int m_index = m * row + i;
            int n_index = n * i + column;
            P[index] += M[m_index] * N[n_index];
        }
    }
}

#define MATRIX_TILE_WIDTH 16
__global__ void mmultiSharedMemoryKernel(float *P, const float *M, const float *N, const int l, const int m, const int n) {
    __shared__ float M_tile[MATRIX_TILE_WIDTH][MATRIX_TILE_WIDTH];
    __shared__ float N_tile[MATRIX_TILE_WIDTH][MATRIX_TILE_WIDTH];

    // Compute output matrix row and column
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * n + column;


    float result = 0;

    // Phase tile around and load elements
    for (int tile_index = 0; tile_index < (m + MATRIX_TILE_WIDTH - 1) / MATRIX_TILE_WIDTH; tile_index++) {
        // Check we have valid index in M tile and load M tile element
        if (row < l && (tile_index * MATRIX_TILE_WIDTH +  threadIdx.x) < m) {
            M_tile[threadIdx.y][threadIdx.x] = M[row * m + tile_index * MATRIX_TILE_WIDTH + threadIdx.x];
        } else {
            M_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Check we have a valid index in N tile and load N tile element
        if (column < n && (tile_index * MATRIX_TILE_WIDTH + threadIdx.y) < m) {
            N_tile[threadIdx.y][threadIdx.x] = N[(tile_index * MATRIX_TILE_WIDTH + threadIdx.y) * n + column];
        } else {
            N_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        if (row < l && column < n) {
            for (int i = 0; i < MATRIX_TILE_WIDTH; i++) {
                result += M_tile[threadIdx.y][i] * N_tile[i][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Check we have valid indicies in the output matrix
    if (row < l && column < n) {
        P[index] = result;
    }
}

#define IN_MATRIX_L 8
#define IN_MATRIX_M 8
#define IN_MATRIX_N 8

#define M_TILE_COLUMNS 8
#define M_TILE_ROWS 4
#define N_TILE_COLUMNS M_TILE_ROWS
#define N_TILE_ROWS M_TILE_COLUMNS

#define NUM_THREADS_X 8
#define NUM_THREADS_Y 4


/*
 * This kernel is used to perform matrix multiplication using shared memory. It uses statically allocated device memory.
 *
 * Supports:
 *   - Shared memory tiles with different row and column dimensions
 *      * Tile dimensions are specified in terms of the M matrix tile via M_TILE_COLUMNS and M_TILE_ROWS, N_TILE_COLUMNS
 *        and N_TILE_ROWS are then defined from there
 *   - Shared memory tiles where the thread block size does not match the tile size
 *      * Tiles can be smaller than the thread block size which means that not all the threads can be used to
 *        cooperatively load a tile. This also results in additional control divergence as some threads will not have
 *        valid input data to compute an output result at all times.
 *      * If tiles are larger than the thread block size, then a single thread needs to load multiple input matrix
 *        values and compute multiple output matrix values.
 */
__global__ void mmultiTiledKernel(float *P, const float *M, const float *N, const int l, const int m, const int n) {
    __shared__ float M_tile[M_TILE_ROWS][M_TILE_COLUMNS];
    __shared__ float N_tile[N_TILE_ROWS][N_TILE_COLUMNS];
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;


    if (blockWidth > M_TILE_COLUMNS || blockHeight > M_TILE_ROWS) {
        // Not enough space for all threads to run cooperatively out of the shared memory

    } if (blockWidth < M_TILE_COLUMNS || blockHeight < M_TILE_ROWS) {
        // Each thread is responsible for more than one output element

    } else {
        // Nominal case: Tile size equals thread block size
        int output_row = blockHeight * blockIdx.y + threadIdx.y;
        int output_column = blockWidth * blockIdx.x + threadIdx.x;
        int index = output_row * n + output_column;
        float result = 0;

        // Phase tile around and load elements. The number of phases is determined by the m (inner) dimension.
        int num_phases = (m + M_TILE_COLUMNS - 1) / M_TILE_COLUMNS;

        for (int tile_index = 0; tile_index < num_phases; tile_index++) {
            // Check we have valid index in M tile and load M tile element
            if (output_row < l && (tile_index * M_TILE_COLUMNS + threadIdx.x) < m) {
                M_tile[threadIdx.y][threadIdx.x] = M[output_row * m + tile_index * M_TILE_COLUMNS + threadIdx.x];
            } else {
                M_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            // Check we have a valid index in N tile and load N tile element
            if (output_column < n && (tile_index * N_TILE_ROWS + threadIdx.x) < m) {

                int N_index = (tile_index * N_TILE_ROWS + threadIdx.x) * n +  blockWidth * blockIdx.x + threadIdx.y;
                printf("Thread X: %d, Y: %d, Block: X %d, Y %d, N Index: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, N_index);
                N_tile[threadIdx.x][threadIdx.y] = N[N_index];
            } else {
                N_tile[threadIdx.x][threadIdx.y] = 0.0f;
            }
            __syncthreads();

            if (output_row < l && output_column < n) {
                for (int i = 0; i < M_TILE_COLUMNS; i++) {
                    result += M_tile[threadIdx.y][i] * N_tile[i][threadIdx.x];
                }
            }
            __syncthreads();
        }

        // Check we have valid indices in the output matrix
        if (output_row < l && output_column < n) {
            P[index] = result;
        }
    }
}



void print_matrix(const float *M, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f, ", M[n * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void TestMatrixMultiplication() {

    float M[]{1,
              2,                 3, 4, 5, 6, 7, 8, 9};
    float N[]{1, 2, 3,    4, 5, 6, 7, 8, 9};
    float P[9]{};

    matrix_multiplication(M, N, P, 2, 1, 3);
    print_matrix(P, 2, 3);
    printf("\n");

    matrix_multiplication(M, N, P, 3, 2, 3);
    print_matrix(P, 3, 3);
    printf("\n");

    matrix_multiplication(M, N, P, 3, 3, 3);
    print_matrix(P, 3, 3);
}

enum class MMKernel {
    Naive,
    Shared
};

cudaError_t runMatrixMultiplyKernel(MMKernel type, float *P, const float *M, const float *N, const int l, const int m, const int n) {

    float *dev_M;
    float *dev_N;
    float *dev_P;

    // Allocate memory
    checkCuda(cudaMalloc((void **) &dev_M, sizeof(float) * l * m));
    checkCuda(cudaMalloc((void **) &dev_N, sizeof(float) * m * n));
    checkCuda(cudaMalloc((void **) &dev_P, sizeof(float) * l * n));

    // Initial device memory
    checkCuda(cudaMemcpy(dev_M, M, sizeof(float) * l * m, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_N, N, sizeof(float) * m * n, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 numThreads{NUM_THREADS_X,  NUM_THREADS_Y, 1};
    dim3 numBlocks{static_cast<uint32_t>((n + NUM_THREADS_X - 1) / NUM_THREADS_X),
                   static_cast<uint32_t>(l + NUM_THREADS_Y - 1) / NUM_THREADS_Y,
                   1};

    printf("Input Matrix Dimensions: (%d x %d) and (%d x %d)\n", l, m, m, n);
    printf("Block Dimensions: (%d x %d)\n", numThreads.y, numThreads.x);
    printf("Grid Dimensions: (%d x %d)\n\n", numBlocks.y, numBlocks.x);

    if (type == MMKernel::Naive) {
        mmultiKernel<<<numBlocks, numThreads>>>(dev_P, dev_M, dev_N, l, m, n);
    } else if (type == MMKernel::Shared) {
        mmultiTiledKernel<<<numBlocks, numThreads>>>(dev_P, dev_M, dev_N, l, m, n);
    }

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // Copy device output buffers to host
    checkCuda(cudaMemcpy(P, dev_P, sizeof(float) * l * n, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(dev_M);
    cudaFree(dev_N);
    cudaFree(dev_P);

    return cudaSuccess;
}

void init_matrix(float *M, const int m, const int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            //M[i * n + j] = static_cast<float>(i + 1) / static_cast<float>(j + 1);
            M[i * n + j] = i * m + j + 1;
        }
    }
}

void init_identity_matrix(float *M, const int m, const int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                M[i * n + j] = 1.0f;
            else
                M[i*n +j ] = 0.0f;
        }
    }
}

bool compare_floats(float x, float y, float absTol) {
    if (std::abs(x - y) <= absTol) {
        return true;
    } else {
        //printf("Check failed: %f, %f\n", x, y);
        return false;
    }
}


bool matrix_equality(float *M, float *N, const int m, const int n, bool verbose) {
    bool equal = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            bool result = compare_floats(M[i * m + j], N[i * m + j], .1);
            equal &= result;
            if (result == false && verbose == true) {
                printf("Row: %4d, Column: %4d,  %10f, %10f\n", i, j, M[i * m + j], N[i * m + j]);
            }
        }
    }
    return equal;
}


float matrix_max_error(float* M, float* N, const int m, const int n) {
    float max_error = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float error = (M[i * m + j] - N[i * m + j]) / M[i * m + j];
            max_error = (error > max_error) ? error : max_error;
        }
    }
    return max_error;
}


int main(void) {

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    //// Single bifurcation.
    //cudaStatus = runBifucationKernel(BifurcationType::Single, 1024);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "Bifurcation failed!");
    //    return 1;
    //}
    //
    //// N bifurcation.
    //cudaStatus = runBifucationKernel(BifurcationType::N, 1024);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "Bifurcation failed!");
    //    return 1;
    //}

    TestMatrixMultiplication();

    int l = IN_MATRIX_L;
    int m = IN_MATRIX_M;
    int n = IN_MATRIX_N;
    size_t size_M = l * m * sizeof(float);
    size_t size_N = m * n * sizeof(float);
    float *M = (float *) malloc(size_M);
    float *N = (float *) malloc(size_N);
    float *P = (float *) malloc(l * n * sizeof(float));
    auto P_shared = (float *) malloc(l * n * sizeof(float));
    float *P_host = (float *) malloc(l * n * sizeof(float));
    init_matrix(M, l, m);
    init_identity_matrix(N, m, n);

    //print_matrix(P_host, l, n);

    //runMatrixMultiplyKernel(MMKernel::Naive, P, M, N, l, m, n);
    //print_matrix(P, l, n);

    runMatrixMultiplyKernel(MMKernel::Shared, P_shared, M, N, l, m, n);
    //print_matrix(P_shared, l, n);

    matrix_multiplication(M, N, P_host, l, m, n);

    //printf("Checking equality for naive implementation.\n");
    //bool ret = false;//matrix_equality(P, P_host, l, n);
    //float error_0 = matrix_max_error(P_host, P, l, n);
    //if (ret == true) {
    //    printf("No difference found.\n");
    //}
    //printf("Max error for naive implementation: %f\n", error_0);

    printf("Checking equality for shared memory implementation.\n");
    bool ret = matrix_equality(P_host, P_shared, l, n, true);
    float error_1 = matrix_max_error(P_host, P_shared, l, n);
    if (ret == true) {
        printf("No difference found.\n");
    }
    printf("Max error for shared mem implementation: %f\n", error_1);


    free(M);
    free(N);
    free(P);
    free(P_host);
    free(P_shared);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
