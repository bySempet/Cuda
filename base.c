#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/Ejercicio2CPU.h"

double timing_CPU(struct timespec begin, struct timespec end){
return((end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0));
}
void gen_matrixes(int a, int b, int c, float *A, float *B, float *C) {
    srand(time(0));
    for(int i = 0; i < a * b; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for(int i = 0; i < b * c; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }
    for(int i = 0; i < a * c; i++) {
        C[i] = (float)rand() / RAND_MAX;
    }
}

double matrix_mul_and_add(int a, int b, int c, float *A, float *B, float *C, float *D) {
    
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    for(int i = 0; i < a; i++) {
        for(int j = 0; j < c; j++) {
            D[i * c + j] = C[i * c + j];
            for(int k = 0; k < b; k++) {
                D[i * c + j] += A[i * b + k] * B[k * c + j];
            }
        }
    }

