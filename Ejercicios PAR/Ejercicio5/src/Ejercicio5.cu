#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THR_PER_BLOCK 1024 

#define BLOCK_SIZE 16 // Ajustar según la memoria de la GPU

__global__ void muladd(float *A, float *B, float *C, int N) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (N / BLOCK_SIZE); ++m) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = A[row * N + (m * BLOCK_SIZE + col)];
        Bs[row][col] = B[(m * BLOCK_SIZE + row) * N + blockCol * BLOCK_SIZE + col];

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        __syncthreads();
    }

    C[(blockRow * BLOCK_SIZE + row) * N + blockCol * BLOCK_SIZE + col] = Cvalue;
}

 double matrix_mul_and_add_gpu(int a, int b, int c, float *A, float *B, float *C, float *D) {
    cudaEvent_t start, stop;
    float *d_A, *d_B, *d_C, *d_D;
    float miliseconds = 0;
    int thr_per_blk, blk_in_grid_x, blk_in_grid_y;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Tamaño de los bloques que caben en la memoria de la GPU
    const int BLOCK_SIZE = ...;

    // Reserva de memoria en la GPU para los bloques
    cudaMalloc(&d_A, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_B, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_C, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&d_D, BLOCK_SIZE * sizeof(float));

    // Número de bloques en cada dimensión
    int numBlocksA = (a + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksB = (b + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksC = (c + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Bucles para procesar cada bloque
    for (int blockA = 0; blockA < numBlocksA; ++blockA) {
        for (int blockB = 0; blockB < numBlocksB; ++blockB) {
            for (int blockC = 0; blockC < numBlocksC; ++blockC) {
                // Cargar bloques en la memoria de la GPU
                cudaMemcpy(d_A, A + blockA * BLOCK_SIZE, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, B + blockB * BLOCK_SIZE, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_C, C + blockC * BLOCK_SIZE, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

                // Realizar la operación de multiplicación y suma
                matrixMulAndAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_D);

                // Guardar el resultado en la matriz D
                cudaMemcpy(D + blockA * BLOCK_SIZE, d_D, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
    }

    // Liberar la memoria de la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return miliseconds;
}