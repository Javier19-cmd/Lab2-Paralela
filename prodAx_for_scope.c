#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void Ax_b(int m, int n, double* A, double* x, double* b) {
    int i, j;
    #pragma omp parallel for num_threads(4) shared(m, n, A, x, b) private(i, j)
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            b[i] += A[i * n + j] * x[j]; // Producto punto
        }
    } /* −−−Fin de parallel for−−− */
}

int main() {
    int m = 10000; // Número de filas de la matriz A y elementos en el vector b
    int n = 10000; // Número de columnas de la matriz A y elementos en el vector x

    double *A = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(m * sizeof(double));

    // Inicialización de matriz A y vector x (aquí llenamos con valores aleatorios)
    for (int i = 0; i < m * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }

    for (int j = 0; j < n; j++) {
        x[j] = (double)rand() / RAND_MAX;
    }

    // Inicialización de vector b
    for (int i = 0; i < m; i++) {
        b[i] = 0.0;
    }

    // Tiempo de inicio
    clock_t start = clock();

    Ax_b(m, n, A, x, b); // Llama a la función

    // Tiempo de finalización
    clock_t end = clock();

    // Calcula el tiempo de ejecución en segundos
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Imprime algunos valores del vector resultado b
    printf("Resultado b:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", b[i]);
    }
    printf("\n");

    // Imprime el tiempo de ejecución
    printf("Tiempo de ejecución: %f segundos\n", cpu_time_used);

    free(A);
    free(x);
    free(b);

    return 0;
}
