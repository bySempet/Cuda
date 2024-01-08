#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    clock_gettime(CLOCK_MONOTONIC, &end);
	return(timing_CPU(begin, end));
}

int main(int argc, char *argv[]) {
    
    
    if(argc !=4){	
        fprintf(stderr, "Error en los argumentos\n");
	    return(0);
    }

    int a , b , c;
    double time_cpu, time_gpu;
    a = strtoul(argv[1], NULL, 10);
    b = strtoul(argv[2], NULL, 10);
    c = strtoul(argv[3], NULL, 10);

    float *A = (float *)malloc(a * b * sizeof(float));
    float *B = (float *)malloc(b * c * sizeof(float));
    float *C = (float *)malloc(a * c * sizeof(float));
    float *D = (float *)malloc(a * c * sizeof(float));

    gen_matrixes(a, b, c, A, B, C);
    time_cpu = matrix_mul_and_add(a, b, c, A, B, C, D);
    time_gpu = matrix_mul_and_add_gpu(a, b, c, A, B, C, D);
    for(int i = 0; i < a; i++) {
        for(int j = 0; j < c; j++) {
            printf("%f ", D[i * c + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}