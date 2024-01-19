#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/Ejercicio5GPU.cuh"

#define THR_PER_BLOCK 1024

___global__ void matrixMulAdd(float *A, float *B, float *C, float *D, int a, int b, int c) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / c;
    int col = idx % c;

    __shared__ float As[THR_PER_BLOCK];
    __shared__ float Bs[THR_PER_BLOCK];

    float sum = 0;

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
        D[row * c + col] = sum + C[row * c + col];
    }
}
 double matrix_mul_and_add_gpu(int a, int b, int c, float *A, float *B, float *C)
{
    cudaEvent_t start, stop;
    float miliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int subSize = 32; 


    int numSubs = (a * c) / (subSize * subSize);


    float *d_A_sub, *d_B_sub, *d_C_sub;
    gpuErrchk(cudaMalloc(&d_A_sub, subSize * subSize * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B_sub, subSize * subSize * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C_sub, subSize * subSize * sizeof(float)));

    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((float)(subSize * subSize) / thr_per_blk);

   for (int i = 0; i < numSubs; ++i) {

        gpuErrchk(cudaMemcpy(d_A_sub, &A[i * subSize * subSize], subSize * subSize * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_B_sub, &B[i * subSize * subSize], subSize * subSize * sizeof(float), cudaMemcpyHostToDevice));

        gpuErrchk(cudaEventRecord(start));

        matrixMulAdd<<<blk_in_grid,thr_per_blk>>>(d_A_sub, d_B_sub, d_C_sub, subSize, subSize, subSize);
        gpuErrchk(cudaEventRecord(stop));

   
        gpuErrchk(cudaMemcpy(&C[i * subSize * subSize], d_C_sub, subSize * subSize * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);


    cudaFree(d_A_sub);
    cudaFree(d_B_sub);
    cudaFree(d_C_sub);

    return (miliseconds);
}
 
