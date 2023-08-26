#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void prodAx(int m, int n, double * restrict A, double * restrict x, double * restrict b);

int main(int argc, char *argv[]) {
    double *A, *x, *b;
    int i, j, m, n;

    m = 10000; // número de filas de la matriz A y elementos en el vector b
    n = 10000; // número de columnas de la matriz A y elementos en el vector x

    // ---- Asignación de memoria para la matriz A ----
    if ((A = (double *)malloc(m * n * sizeof(double))) == NULL)
        perror("memory allocation for A");

    // ---- Asignación de memoria para el vector x ----
    if ((x = (double *)malloc(n * sizeof(double))) == NULL)
        perror("memory allocation for x");

    // ---- Asignación de memoria para el vector b ----
    if ((b = (double *)malloc(m * sizeof(double))) == NULL)
        perror("memory allocation for b");

    // Inicialización de matriz A con valores aleatorios entre 0 y 1
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A[i * n + j] = (double)rand() / RAND_MAX;

    // Inicialización del vector x con valores aleatorios entre 0 y 1
    for (j = 0; j < n; j++)
        x[j] = (double)rand() / RAND_MAX;

    printf("Calculando el producto Ax para m = %d n = %d\n", m, n);

    clock_t start = clock(); // Marca el inicio del tiempo de ejecución
    prodAx(m, n, A, x, b);
    clock_t end = clock(); // Marca el final del tiempo de ejecución

    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nb:\n");
    for (j = 0; j < n; j++)
        printf("\t%0.6f ", b[j]);
    printf("\n\n");

    printf("Tiempo de ejecución: %f segundos\n", cpu_time_used);

    free(A);
    free(x);
    free(b);

    return 0;
}

/* ------------------------
 * prodAx
 * ------------------------
 */
void prodAx(int m, int n, double * restrict A, double * restrict x, double * restrict b) {
    int i, j;

    for (i = 0; i < m; i++) {
        b[i] = 0.0;

        for (j = 0; j < n; j++) {
            b[i] += A[i * n + j] * x[j];
        }
    }
} //----prodAx----
