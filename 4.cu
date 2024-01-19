#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define THR_PER_BLOCK 1024 
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

__global__ void matrixMulAdd(float *A, float *B, float *C, float *D, int a, int b, int c) {
     int warpId = threadIdx.x / WARP_SIZE;


    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    
    for (int i = 0; i < b; i += WMMA_K) {

        wmma::load_matrix_sync(a_frag, A + (blockIdx.x * WMMA_M + warpId * WMMA_M/2) * b + i, b);
        wmma::load_matrix_sync(b_frag, B + i * c + blockIdx.y * WMMA_N, c);


        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
 wmma::load_matrix_sync(d_frag, C + blockIdx.x * WMMA_M * c + blockIdx.y * WMMA_N, c, wmma::mem_row_major);

    
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = c_frag.x[i] + d_frag.x[i];
    }

   
    wmma::store_matrix_sync(D + blockIdx.x * WMMA_M * c + blockIdx.y * WMMA_N, c_frag, c, wmma::mem_row_major);
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
    blk_in_grid = ceil ((float)(a*c) / thr_per_blk);

   gpuErrchk(cudaEventRecord(start));
    matrixMulAdd<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, d_D, a, b, c);
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



