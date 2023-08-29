#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

void Ax_b(int m, int n, double *A, double *x, double *b, char *strategy, int block_size)
{
    int i, j;

    if (strcmp(strategy, "static") == 0)
    {
#pragma omp parallel for num_threads(4) shared(m, n, A, x, b) private(i, j) schedule(static, block_size)
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                b[i] += A[i * n + j] * x[j]; // Producto punto
            }
        }
    }
    else if (strcmp(strategy, "dynamic") == 0)
    {
#pragma omp parallel for num_threads(4) shared(m, n, A, x, b) private(i, j) schedule(dynamic, block_size)
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                b[i] += A[i * n + j] * x[j]; // Producto punto
            }
        }
    }
    else if (strcmp(strategy, "guided") == 0)
    {
#pragma omp parallel for num_threads(4) shared(m, n, A, x, b) private(i, j) schedule(guided, block_size)
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                b[i] += A[i * n + j] * x[j]; // Producto punto
            }
        }
    }
}

int main()
{
    int m = 10000; // Número de filas de la matriz A y elementos en el vector b
    int n = 10000; // Número de columnas de la matriz A y elementos en el vector x

    double *A = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(m * sizeof(double));

    FILE *file;

    // Intentar leer matriz A desde archivo
    file = fopen("matrix.txt", "r");
    if (file)
    {
        for (int i = 0; i < m * n; i++)
        {
            fscanf(file, "%lf", &A[i]);
        }
        fclose(file);
    }
    else
    {
        // Generación aleatoria si no se encuentra el archivo
        for (int i = 0; i < m * n; i++)
        {
            A[i] = (double)rand() / RAND_MAX;
        }
    }

    // Intentar leer vector x desde archivo
    file = fopen("vector.txt", "r");
    if (file)
    {
        for (int j = 0; j < n; j++)
        {
            fscanf(file, "%lf", &x[j]);
        }
        fclose(file);
    }
    else
    {
        // Generación aleatoria si no se encuentra el archivo
        for (int j = 0; j < n; j++)
        {
            x[j] = (double)rand() / RAND_MAX;
        }
    }

    // Imprimir encabezados
    printf("| %-10s | %-10s | %-20s |\n", "Estrategia", "Bloque", "Tiempo (segundos)");
    printf("|------------|------------|---------------------|\n");

    // Estrategias y tamaños de bloque
    char *strategies[] = {"static", "dynamic", "guided"};
    char *strategy_names[] = {"Static", "Dynamic", "Guided"};
    int block_sizes[3][3] = {{100000, 10000, 1000}, {100000, 10000, 1000}, {1000, 100, 10}};

    // Iterar sobre cada estrategia y tamaño de bloque
    for (int s = 0; s < 3; s++)
    {
        for (int blk = 0; blk < 3; blk++)
        {
            // Reiniciar el vector b a cero
            for (int idx = 0; idx < m; idx++)
            {
                b[idx] = 0.0;
            }

            // Registrar el tiempo de inicio
            double start_time = omp_get_wtime();
            // Llamar a la función con la estrategia y tamaño de bloque actual
            Ax_b(m, n, A, x, b, strategies[s], block_sizes[s][blk]);
            // Registrar el tiempo de finalización
            double end_time = omp_get_wtime();

            // Calcular e imprimir el tiempo de ejecución
            printf("| %-10s | %-10d | %-20f |\n", strategy_names[s], block_sizes[s][blk], end_time - start_time);
        }
    }

    free(A);
    free(x);
    free(b);

    return 0;
}
