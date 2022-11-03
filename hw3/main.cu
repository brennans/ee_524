#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#include <check_cuda.h>


__global__ void singleBifurcationKernel(int *b, const int *a, const int index) {
    int i = threadIdx.x;
    if (i < index) {
        b[i] = a[i];
    } else {
        b[i] = 0;
    }
}

__global__ void nBifurcationKernel(int *b, const int *a) {
    int n = threadIdx.x;
    for (int i = 0; i < n; i++) {
        b[i] = a[i] + a[i];
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
        singleBifurcationKernel<<<1, size>>>(dev_b, dev_a, size / 2);
    } else if (BifurcationType::N == type) {
        nBifurcationKernel<<<1, size>>>(dev_b, dev_a);
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
            int out_index = i * l + j;
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

#define MATRIX_TILE_WIDTH 32
__global__ void mmultiSharedMemoryKernel(float *P, const float *M, const float *N, const int l, const int m, const int n) {
    __shared__ float M_tile[MATRIX_TILE_WIDTH][MATRIX_TILE_WIDTH];
    __shared__ float N_tile[MATRIX_TILE_WIDTH][MATRIX_TILE_WIDTH];

    // Compute output matrix row and column
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * n + column;

    // Check we have valid indicies in the output matrix
    if (row < l && column < n) {
        float result = 0;

        // Move tile around and load elements
        for (int tile_index = 0; tile_index < ceil(m / (float) MATRIX_TILE_WIDTH); tile_index++) {
            // Check we have valid index in M tile and load M tile element
            if ((tile_index * MATRIX_TILE_WIDTH +  threadIdx.x) < m) {
                M_tile[threadIdx.y][threadIdx.x] = M[row * m + tile_index * MATRIX_TILE_WIDTH + threadIdx.x];
            } else {
                M_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            // Check we have a valid index in N tile and load N tile element
            if ((tile_index * MATRIX_TILE_WIDTH + threadIdx.y) < m) {
                N_tile[threadIdx.y][threadIdx.x] = N[(tile_index * MATRIX_TILE_WIDTH + threadIdx.y) * n + column];
            } else {
                N_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();

            for (int i = 0; i < MATRIX_TILE_WIDTH; i++) {
                result += M_tile[threadIdx.y][i] * N_tile[i][threadIdx.x];
            }
            __syncthreads();
        }
        P[index] = result;
    }
}

void print_matrix(const float *M, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f, ", M[m * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void TestMatrixMultiplication() {

    float M[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    float N[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    float P[9]{};

    matrix_multiplication(M, N, P, 2, 3, 2);
    print_matrix(P, 2, 2);
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
    int TILE_SIZE = 32;
    dim3 numThreads{32, 32, 1};
    dim3 numBlocks{static_cast<uint32_t>((l + TILE_SIZE - 1) / TILE_SIZE),
                   static_cast<uint32_t>(n + TILE_SIZE - 1) / TILE_SIZE,
                   1};
    if (type == MMKernel::Naive) {
        mmultiKernel<<<numBlocks, numThreads>>>(dev_P, dev_M, dev_N, l, m, n);
    } else if (type == MMKernel::Shared) {
        mmultiSharedMemoryKernel<<<numBlocks, numThreads>>>(dev_P, dev_M, dev_N, l, m, n);
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
            M[i * m + j] = static_cast<float>(i + 1) / static_cast<float>(j + 1);
        }
    }
}

bool compare_floats(float x, float y, float absTol) {
    if (std::abs(x - y) <= absTol) {
        return true;
    } else {
        printf("Check failed: %f, %f\n", x, y);
        return false;
    }
}


bool matrix_equality(float *M, float *N, const int m, const int n) {
    bool equal = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            equal &= compare_floats(M[i * m + j], N[i * m + j], .1);
        }
    }
    return equal;
}


int main(void) {

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }

    // Single bifurcation.
    cudaStatus = runBifucationKernel(BifurcationType::Single, 128);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Bifurcation failed!");
        return 1;
    }

    // N bifurcation.
    cudaStatus = runBifucationKernel(BifurcationType::N, 128);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Bifurcation failed!");
        return 1;
    }

    TestMatrixMultiplication();

    int l = 1000;
    int m = 1000;
    int n = 1000;
    float *M = (float *) malloc(l * m * sizeof(float));
    float *N = (float *) malloc(m * n * sizeof(float));
    float *P = (float *) malloc(l * n * sizeof(float));
    float *P_shared = (float *) malloc(l * n * sizeof(float));
    float *P_host = (float *) malloc(l * n * sizeof(float));
    init_matrix(M, l, m);
    init_matrix(N, m, n);
    runMatrixMultiplyKernel(MMKernel::Naive, P, M, N, l, m, n);
    //print_matrix(P, l, n);

    matrix_multiplication(M, M, P_host, l, m, n);
    //print_matrix(P_host, l, n);

    runMatrixMultiplyKernel(MMKernel::Shared, P_shared, M, N, l, m, n);
    //print_matrix(P_shared, l, n);

    printf("Checking equality for naive implementation.\n");
    bool ret = matrix_equality(P, P_host, l, n);
    if (ret == true) {
        printf("No difference found.\n");
    }

    printf("Checking equality for shared memory implementation.\n");
    ret = matrix_equality(P_shared, P_host, l, n);
    if (ret == true) {
        printf("No difference found.\n");
    }

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
