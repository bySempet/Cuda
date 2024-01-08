#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <../include/Ejercicio3.h>
#include <../include/Ejercicio3.cuh>

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