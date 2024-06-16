#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "../include/Ejercicio4GPU.cuh"

#define THR_PER_BLOCK 1024  
#define WMMA_GLOBAL 16
#define CUADRADO 256
#define WARP_SIZE 32
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

using namespace nvcuda;

__global__ void matrixMulAdd(half *A, half *B, float *C, const int M, const int K, const int N) {

// Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;
     
   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

// Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;


  wmma::fill_fragment(acc_frag, 0.0f);

// Loop over the K-dimension
   for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

   // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
       // Load the inputs
       wmma::load_matrix_sync(a_frag, A + aRow + aCol * lda, lda);
       wmma::load_matrix_sync(b_frag, B + bRow + bCol * ldb, ldb);
 
       // Perform the matrix multiplication
       wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
   }

  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
 
  if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(c_frag, C + cRow + cCol * ldc, ldc, wmma::mem_col_major);

    #pragma unroll
    for(int i=0; i < c_frag.num_elements; i++) {
       c_frag.x[i] = 1.0f * acc_frag.x[i] + 1.0f * c_frag.x[i];
    }
           // Store the output
       wmma::store_matrix_sync(C + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

double matrix_mul_and_add_gpu(int a, int b, int c, float *A, float *B, float *C, float *D) {
    
   cudaEvent_t start, stop;
   float  *d_C;
   half *d_A, *d_B;
   float miliseconds = 0;

   int valor_peque_a = (a + WMMA_GLOBAL -1)/ WMMA_GLOBAL * WMMA_GLOBAL;
   int valor_peque_b = (b + WMMA_GLOBAL -1)/ WMMA_GLOBAL * WMMA_GLOBAL;
   int valor_peque_c = (c + WMMA_GLOBAL -1)/ WMMA_GLOBAL * WMMA_GLOBAL;

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   half *a_peque = (half *) malloc(valor_peque_a * valor_peque_b * sizeof(half));
   half *b_peque = (half *) malloc(valor_peque_c * valor_peque_b * sizeof(half));
   float *c_peque = (float *) malloc(valor_peque_a * valor_peque_c * sizeof(float));

   for( int i = 0; i < a; i++)
   {
       for(int j= 0; j < b; j++)
       {
         a_peque[i * valor_peque_b + j] = __float2half(A[i*b+j]);
       }
   }

   for( int i = 0; i < b; i++)
   {
       for(int j= 0; j < c; j++)
       {
         b_peque[i * valor_peque_c + j] = __float2half(B[i*c+j]);
       }
   }


  for( int i = 0; i < a; i++)
   {
       for(int j= 0; j < c; j++)
       {
         c_peque[i * valor_peque_c + j] = C[i*c+j];
       }
   }

   gpuErrchk(cudaMalloc((void **)&d_A, valor_peque_a * valor_peque_b * sizeof(half)));
   gpuErrchk(cudaMalloc((void **)&d_B, valor_peque_b * valor_peque_c * sizeof(half)));
   gpuErrchk(cudaMalloc((void **)&d_C, valor_peque_a * valor_peque_c * sizeof(float)));

   gpuErrchk(cudaMemcpy((void *)d_A, a_peque,valor_peque_a * valor_peque_b * sizeof(half), cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy((void *)d_B, b_peque, valor_peque_b * valor_peque_c * sizeof(half), cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy((void *)d_C, c_peque, valor_peque_a * valor_peque_c * sizeof(float), cudaMemcpyHostToDevice));


   dim3 blockDim(4 * WARP_SIZE,4);
   dim3 gridDim((valor_peque_a + (WMMA_GLOBAL * blockDim.x / WARP_SIZE -1)) / (WMMA_GLOBAL * blockDim.x / WARP_SIZE),(valor_peque_c + WMMA_GLOBAL * blockDim.y -1) / >

   gpuErrchk(cudaEventRecord(start));
   matrixMulAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, valor_peque_a, valor_peque_b, valor_peque_c);
   gpuErrchk(cudaEventRecord(stop));
    
   cudaDeviceSynchronize();

   cudaMemcpy((void *)c_peque, d_C, valor_peque_a * valor_peque_c * sizeof(float), cudaMemcpyDeviceToHost);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&miliseconds, start, stop);


  for (int i = 0; i < a; i++)
       {
               for (int j = 0; j < c; j++)
               {
                 D[i * c + j] = c_peque[i * valor_peque_c + j];
               }
       }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   free(a_peque);
   free(b_peque);
   free(c_peque);
 
   return (miliseconds);
}





