#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/Ejercicio3GPU.cuh"

#define THR_PER_BLOCK 1024 
#define BLOCK_SIZE 1024
__global__ void matrixMulAdd(float *A, float *B, float *C, float *D, int a, int b, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / c;
    int col = idx % c;

    __shared__ float As[THR_PER_BLOCK];
    __shared__ float Bs[THR_PER_BLOCK];
//    __shared__ float Cs[THR_PER_BLOCK];

    float sum = 0.0;
         if (idx > a * c) {
        return; 
    }
    for (int sub = 0; sub < (b + THR_PER_BLOCK - 1) / THR_PER_BLOCK; ++sub) {
        if (row < a && sub * THR_PER_BLOCK + threadIdx.x < b) {
            As[threadIdx.x] = A[row * b + sub * THR_PER_BLOCK + threadIdx.x];
        } else {
            As[threadIdx.x] = 0.0;
        }

        if (sub * THR_PER_BLOCK + threadIdx.x < b && col < c) {
            Bs[threadIdx.x] = B[(sub * THR_PER_BLOCK + threadIdx.x) * c + col];
        } else {
            Bs[threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int i = 0; i < THR_PER_BLOCK; ++i) {
            sum += As[i] * Bs[i];
        }

        __syncthreads();
    }

    if(row < a && col < c) {
        D[row * c + col] = sum + C[row *blockDim.x + col];
    }
}


 double matrix_mul_and_add_gpu(int a, int b, int c, float *A, float *B, float *C, float *D) {
    
    cudaEvent_t start, stop;
    float *d_A, *d_B, *d_C, *d_D;
    float miliseconds = 0;
    int thr_per_blk, blk_in_grid;
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
    blk_in_grid = ceil((float)(a * c) / thr_per_blk);

    gpuErrchk(cudaEventRecord(start));
    matrixMulAdd<<<blk_in_grid,thr_per_blk>>>(d_A, d_B, d_C, d_D, a, b, c);
    gpuErrchk(cudaEventRecord(stop));
   
    cudaMemcpy(D, d_D, a * c * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return (miliseconds);
}
Tiempo CPU: 0.000005
Tiempo GPU: 0.019776
