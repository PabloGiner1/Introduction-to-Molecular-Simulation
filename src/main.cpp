#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    int model;
    int algorithm;

    unsigned int seed;

    SimulationParameters simulation;
    ModelParameters modelParameters;

    char algorithmText[50];
    char algorithmName[30];
    char outputFile[150];

    char line[200];
    char name[50];
    char value[100];

    FILE *configFile;

    // Elegimos el modelo físico
    model = HARMONIC;

    // Parámetros que se mantienen fijos
    simulation.mass = 1.0;
    simulation.kBT = 1.0;

    simulation.initialPosition = 1.0;
    simulation.initialVelocity = 0.0;

    modelParameters.k = 1.0;

    // Valores por defecto
    algorithm = EULER_MARUYAMA;
    strcpy(algorithmText, "EULER_MARUYAMA");

    simulation.gamma = 1.0;
    simulation.dt = 0.001;
    simulation.totalTime = 1000.0;

    seed = 12345;

    // Abrimos el archivo de configuración
    configFile = fopen("config/simulation.txt", "r");

    if (configFile == NULL)
    {
        printf("Error opening config/simulation.txt\n");
        return 1;
    }

    // Leemos los parámetros
    while (fgets(line, sizeof(line), configFile) != NULL)
    {
        if (line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        if (sscanf(line, " %49[^=]=%99s", name, value) == 2)
        {
            if (strcmp(name, "algorithm") == 0)
            {
                strcpy(algorithmText, value);

                if (strcmp(value, "EULER_MARUYAMA") == 0)
                {
                    algorithm = EULER_MARUYAMA;
                }

                else if (strcmp(value, "RUNGE_KUTTA") == 0)
                {
                    algorithm = RUNGE_KUTTA;
                }

                else if (strcmp(value, "GJF") == 0)
                {
                    algorithm = GJF;
                }

                else
                {
                    printf("Error: unknown algorithm\n");
                    fclose(configFile);
                    return 1;
                }
            }

            else if (strcmp(name, "eta") == 0)
            {
                simulation.gamma = atof(value);
            }

            else if (strcmp(name, "h") == 0)
            {
                simulation.dt = atof(value);
            }

            else if (strcmp(name, "totalTime") == 0)
            {
                simulation.totalTime = atof(value);
            }

            else if (strcmp(name, "seed") == 0)
            {
                seed = (unsigned int)atoi(value);
            }
        }
    }

    fclose(configFile);

    // Inicializamos el generador de números aleatorios
    srand(seed);

    // Asignamos un nombre al algoritmo
    if (algorithm == EULER_MARUYAMA)
    {
        sprintf(algorithmName, "euler");
    }

    if (algorithm == RUNGE_KUTTA)
    {
        sprintf(algorithmName, "rk");
    }

    if (algorithm == GJF)
    {
        sprintf(algorithmName, "gjf");
    }

    // Creamos automáticamente el nombre del archivo
    sprintf(
        outputFile,
        "data/raw/harmonic/harmonic_%s.csv",
        algorithmName);

    // Mostramos los parámetros utilizados
    printf("\n");
    printf("Algorithm: %s\n", algorithmText);
    printf("eta = %g\n", simulation.gamma);
    printf("h = %g\n", simulation.dt);
    printf("Total time = %g\n", simulation.totalTime);
    printf("Seed = %u\n", seed);
    printf("\n");

    // Ejecutamos la simulación
    runSimulation(
        model,
        algorithm,
        simulation,
        modelParameters,
        outputFile);

    printf("Simulation finished\n");
    printf("Results saved in: %s\n", outputFile);

    return 0;
}