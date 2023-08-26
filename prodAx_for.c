#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

void Ax_b(int m, int n, double* A, double* x, double* b) {
    int i, j;
    #pragma omp parallel for
    for(i = 0; i < m; i++) {
        b[i] = 0.0;  // inicialización elemento i del vec.
        for(j = 0; j < n; j++) {
            b[i] += A[i*n + j] * x[j]; // producto punto
        }
    } /* −−−Fin de parallel for−−− */
}

int main() {
    int m = 20000; // número de filas de la matriz A y elementos en el vector b
    int n = 20000; // número de columnas de la matriz A y elementos en el vector x
    
    double *A = (double *)malloc(m * n * sizeof(double)); // matriz A
    double *x = (double *)malloc(n * sizeof(double)); // vector x
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)rand() / RAND_MAX; // llena la matriz A con valores aleatorios entre 0 y 1
        }
    }
    
    for (int j = 0; j < n; j++) {
        x[j] = (double)rand() / RAND_MAX; // llena el vector x con valores aleatorios entre 0 y 1
    }
    
    double b[m]; // vector resultado b
    
    clock_t start = clock(); // Marca el inicio del tiempo de ejecución
    
    Ax_b(m, n, A, x, b); // llama a la función
    
    clock_t end = clock(); // Marca el final del tiempo de ejecución
    
    // Imprime algunos valores del vector resultado b para verificar
    printf("Resultado b:\n");
    for (int i = 0; i < 3; i++) {
        printf("%.6f ", b[i]);
    }
    printf("\n...\n");
    
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución: %f segundos\n", cpu_time_used);
    
    free(A);
    free(x);
    
    return 0;
}
