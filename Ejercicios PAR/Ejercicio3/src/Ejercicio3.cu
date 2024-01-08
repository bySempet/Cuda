#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THR_PER_BLOCK 1024 

__global__ void matrixMulAdd(float *A, float *B, float *C, float *D, int a, int b, int c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Cs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0;

    for (int sub = 0; sub < (b + BLOCK_SIZE - 1) / BLOCK_SIZE; ++sub) {
        if (sub * BLOCK_SIZE + threadIdx.x < b && row < a) {
            As[threadIdx.y][threadIdx.x] = A[row * b + sub * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (sub * BLOCK_SIZE + threadIdx.y < b && col < c) {
            Bs[threadIdx.y][threadIdx.x] = B[(sub * BLOCK_SIZE + threadIdx.y) * c + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < a && col < c) {
        Cs[threadIdx.y][threadIdx.x] = C[row * c + col];
        __syncthreads();
        D[row * c + col] = sum + Cs[threadIdx.y][threadIdx.x];
    }
}


 double matrix_mul_and_add_gpu(int a, int b, int c, float *A, float *B, float *C, float *D) {
    
    cudaEvent_t start, stop;
    float *d_A, *d_B, *d_C, *d_D;
    float miliseconds = 0;
    int thr_per_blk, blk_in_grid_x, blk_in_grid_y;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, a * b * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, b * c * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, a * c * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_D, a * c * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_A, A, a * b * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, b * c * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C, C, a * c * sizeof(float), cudaMemcpyHostToDevice));

    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid_x = (c + thr_per_blk - 1) / thr_per_blk;
    blk_in_grid_y = (a + thr_per_blk - 1) / thr_per_blk;

    gpuErrchk(cudaEventRecord(start));
    matrixMulAdd<<<dim3(blk_in_grid_x, blk_in_grid_y), dim3(thr_per_blk, thr_per_blk)>>>(d_A, d_B, d_C, d_D, a, b, c);
    gpuErrchk(cudaEventRecord(stop));
   
    cudaMemcpy(D, d_D, a * c * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return (miliseconds);
}