#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

void Ax_b(int m, int n, double *A, double *x, double *b)
{
    int i, j;
    int num_threads = omp_get_max_threads();
    double **local_b = malloc(num_threads * sizeof(double *));

    for (int t = 0; t < num_threads; t++)
    {
        local_b[t] = malloc(m * sizeof(double));
        memset(local_b[t], 0, m * sizeof(double));
    }

#pragma omp parallel private(i, j) num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        double *local_b_ptr = local_b[tid];

#pragma omp for schedule(guided, 10)
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                local_b_ptr[i] += A[i * n + j] * x[j];
            }
        }
    }

    // Combina todos los resultados locales en el vector b
    for (int t = 0; t < num_threads; t++)
    {
        for (i = 0; i < m; i++)
        {
            b[i] += local_b[t][i];
        }
        free(local_b[t]);
    }
    free(local_b);
}

int main()
{
    int m = 10000;
    int n = 10000;

    double *A = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(m * sizeof(double));

    FILE *file;

    // Leer matriz A
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
        for (int i = 0; i < m * n; i++)
        {
            A[i] = (double)rand() / RAND_MAX;
        }
    }

    // Leer vector x
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
        for (int j = 0; j < n; j++)
        {
            x[j] = (double)rand() / RAND_MAX;
        }
    }

    // Reiniciar el vector b a cero
    for (int idx = 0; idx < m; idx++)
    {
        b[idx] = 0.0;
    }

    // Registrar tiempo de inicio
    double start_time = omp_get_wtime();
    Ax_b(m, n, A, x, b);
    double end_time = omp_get_wtime();

    printf("+------------+------------+-----------------+\n");
    printf("| Estrategia |   Bloque   |Tiempo (Segundos)|\n");
    printf("+------------+------------+-----------------+\n");
    printf("|   Guided   |     10     | %f segundos |\n", end_time - start_time);
    printf("+------------+------------+-----------------+\n");

    free(A);
    free(x);
    free(b);

    return 0;
}
