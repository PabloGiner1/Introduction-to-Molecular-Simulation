#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int N;
    int i;

    double u;

    FILE *file;

    // Número de valores que vamos a generar
    N = 10000000;

    // Inicializamos rand()
    srand(time(NULL));

    // Abrimos el archivo de salida
    file = fopen(
        "data/raw/random/random_uniform.csv",
        "w");

    if (file == NULL)
    {
        printf("Error opening output file\n");
        return 1;
    }

    fprintf(file, "u\n");

    // Generamos números uniformes entre 0 y 1
    for (i = 0; i < N; i++)
    {
        u = (double)rand() / RAND_MAX;

        fprintf(file, "%.17g\n", u);
    }

    fclose(file);

    printf("Random test completed\n");
    printf("Generated numbers: %d\n", N);
    printf("RAND_MAX = %d\n", RAND_MAX);
    printf(
        "Results saved in: "
        "data/raw/random/random_uniform.csv\n");

    return 0;
}