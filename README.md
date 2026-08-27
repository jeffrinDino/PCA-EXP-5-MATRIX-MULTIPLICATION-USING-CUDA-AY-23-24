# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME: G.V JEFFRIN DINO</h3>
<h3>ENTER YOUR REGISTER NO: 212225040150</h3>
<h3>EX. NO:5</h3>
<h3>DATE: 27.08.2026</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
```python
%%writefile matmul.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define SIZE 4
#define BLOCK_SIZE 2

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", \
                __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

inline double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

// -------------------------------
// CUDA kernel for matrix multiply
// -------------------------------
__global__ void matrixMultiply(const int *a, const int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size)
    {
        int sum = 0;
        for (int k = 0; k < size; ++k)
            sum += a[row * size + k] * b[k * size + col];
        c[row * size + col] = sum;
    }
}

// -------------------------------
// Host code
// -------------------------------
int main()
{
    int a[SIZE][SIZE], b[SIZE][SIZE];
    int c_gpu[SIZE][SIZE], c_cpu[SIZE][SIZE];
    int *dev_a, *dev_b, *dev_c;
    int bytes = SIZE * SIZE * sizeof(int);

    // Initialize input matrices
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j) {
            a[i][j] = i + j;
            b[i][j] = i - j;
            c_gpu[i][j] = 0;
            c_cpu[i][j] = 0;
        }

    // ==============================
    // Host (CPU) matrix multiplication
    // ==============================
    double cpu_start = seconds();
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                c_cpu[i][j] += a[i][k] * b[k][j];
    double cpu_end = seconds();

    // ==============================
    // Device (GPU) matrix multiplication
    // ==============================
    CHECK(cudaMalloc((void**)&dev_a, bytes));
    CHECK(cudaMalloc((void**)&dev_b, bytes));
    CHECK(cudaMalloc((void**)&dev_c, bytes));

    CHECK(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Measure GPU time using CUDA events (accurate GPU timing)
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);
    cudaEventRecord(stop, 0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&gpuTime, start, stop);

    CHECK(cudaMemcpy(c_gpu, dev_c, bytes, cudaMemcpyDeviceToHost));

    // ==============================
    // Print results
    // ==============================
    printf("\nCPU Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            printf("%4d ", c_cpu[i][j]);
        printf("\n");
    }

    printf("\nGPU Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            printf("%4d ", c_gpu[i][j]);
        printf("\n");
    }

    // Verify correctness
    int match = 1;
    for (int i = 0; i < SIZE && match; ++i)
        for (int j = 0; j < SIZE; ++j)
            if (c_cpu[i][j] != c_gpu[i][j]) {
                match = 0;
                break;
            }

    printf("\nVerification: %s\n", match ? "SUCCESS" : "FAIL");

    printf("CPU Time: %.6f seconds\n", cpu_end - cpu_start);
    printf("GPU Time: %.6f milliseconds\n", gpuTime);

    // Cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

```

## OUTPUT:

<img width="250" height="267" alt="image" src="https://github.com/user-attachments/assets/1febd731-74f5-46e0-b572-cfdadcdeb5a6" />

## RESULT:
Thus the program has been executed by using CUDA to mulptiply two matrices. It is observed that there are variations in host and device elapsed time. Device took ______________time and host took ___________time.
