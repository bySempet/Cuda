#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/Ejercicio3GPU.cuh"

#define THR_PER_BLOCK 1024  
#define BLOCK_SIZE 1024
#define TILE_WITDH 32
__global__ void matrixMulAdd(float *A, float *B, float *C, float *D, int a, int b, int c) {

   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   __shared__ float shared_A[TILE_WITDH][TILE_WITDH];
   __shared__ float shared_B[TILE_WITDH][TILE_WITDH];

   float sum = 0.0;

   for (int sub = 0; sub < (b + TILE_WITDH - 1) / TILE_WITDH ; ++sub) {
       if (row < a && sub * TILE_WITDH + threadIdx.x < b) shared_A[threadIdx.y][threadIdx.x] = A[row * b + sub * TILE_WITDH + threadIdx.x];
       else  shared_A[threadIdx.y][threadIdx.x]= 0.0;

       if (sub * TILE_WITDH + threadIdx.y < b && col < c)  shared_B[threadIdx.y][threadIdx.x] = B[(sub * TILE_WITDH + threadIdx.y) * c + col];
       else shared_B[threadIdx.y][threadIdx.x]= 0.0;

       __syncthreads();
       //Aqui da problemas por el tamaño, checkear bucle. Solucionado.Depende de si cuadra o no el tamaño.
       int iteraciones = TILE_WITDH;
       if(sub == (( b * TILE_WITDH -1)/TILE_WITDH -1)) iteraciones = b % TILE_WITDH;
       for (int i = 0; i < iteraciones; ++i) {
           sum += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
       }

  __syncthreads();
   }

   if(row < a && col < c) {
       D[row * c + col] = sum + C[row * c + col];
   }
}


double matrix_mul_and_add_gpu(int a, int b, int c, float *A, float *B, float *C, float *D) {
    
   cudaEvent_t start, stop;
   float *d_A, *d_B, *d_C, *d_D;
   float miliseconds = 0;

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   gpuErrchk(cudaMalloc(&d_A, a * b * sizeof(float)));
   gpuErrchk(cudaMalloc(&d_B, b * c * sizeof(float)));
   gpuErrchk(cudaMalloc(&d_C, a * c * sizeof(float)));
   gpuErrchk(cudaMalloc(&d_D, a * c * sizeof(float)));

   gpuErrchk(cudaMemcpy(d_A, A, a * b * sizeof(float), cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(d_B, B, b * c * sizeof(float), cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(d_C, C, a * c * sizeof(float), cudaMemcpyHostToDevice));

   dim3 blockDim(TILE_WITDH,TILE_WITDH);
   dim3 gridDim((c+blockDim.x -1) /blockDim.x, (a + blockDim.y -1) / blockDim.y);

   gpuErrchk(cudaEventRecord(start));
   matrixMulAdd<<<gridDim,blockDim>>>(d_A, d_B, d_C, d_D, a, b, c);
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
