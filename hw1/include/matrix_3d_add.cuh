#ifndef HW1_MATRIX_3D_ADD_CUH
#define HW1_MATRIX_3D_ADD_CUH

__global__ void matrix_3d_add(int *c, const int* a, const int* b, const int M, const int N, const int P)
{
    int columnIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int rowIndex = blockDim.y * blockIdx.y + threadIdx.y;
    int depthIndex = blockDim.z * blockIdx.z + threadIdx.z;

    if (rowIndex < M && columnIndex < N && depthIndex < P) {
        int index = (depthIndex * M * N ) + (rowIndex * N) + columnIndex;
        c[index] = a[index] + b[index];
    }
}


cudaError_t runMatrix3dAdd(int *c, const int *a, const int *b, uint32_t m, uint32_t n, uint32_t p, uint32_t threadDim)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    uint32_t size = m * n * p;
    dim3 numThreads {threadDim,threadDim,1};
    dim3 numBlocks {(n + threadDim - 1) / threadDim, (m + threadDim - 1) / threadDim, (p + threadDim - 1) / threadDim};


    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCuda(cudaMalloc((void**)&dev_c, size * sizeof(int)));
    checkCuda(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    checkCuda(cudaMalloc((void**)&dev_b, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    checkCuda(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each element.
    matrix_3d_add<<<numBlocks, numThreads>>>(dev_c, dev_a, dev_b, m, n, p);


    // Check for any errors launching the kernel
    cudaStatus = checkCuda(cudaGetLastError());

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCuda(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCuda(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

void checkMatrix3dAdd(const int *c, const int* a, const int* b, const int M, const int N, const int P)
{
    for (int k = 0; k < P; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int index = k * M * N + i * N + j;
                int expected = a[index] + b[index];
                if (expected != c[index]) {
                    int result = c[index];
                    printf("Mismatch between %d, %d at index %d\n", result, c[index], index);
                }
            }
        }
    }

}

#endif //HW1_MATRIX_3D_ADD_CUH
