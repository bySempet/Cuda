#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/Ejercicio2GPU.cuh"

#define THR_PER_BLOCK 1024 

__global__ void matrixMulAdd(float *A, float *B, float *C, float *D, int a, int b, int c) {
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int col = idx / b;
        int row = idx % b;

    if(idx < a * c) {
        float sum = 0;
        for(int i = 0; i < b; i++) {
            sum += A[row * b + i] * B[i * c + col];
        }
        D[row * c + col] = sum + C[row * c + col];
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

Tiempo de la CPU: 5.583517Tiempo de la GPU: 48.012383

