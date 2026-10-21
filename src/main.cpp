#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    int model;
    int algorithm;
    bool useRandomSeed;

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
    model = DOUBLE_WELL;

    // Parámetros fijos
    simulation.mass = 1.0;
    simulation.kBT = 0.2;

    simulation.initialPosition = 1.0;
    simulation.initialVelocity = 0.0;

    modelParameters.k = 1.0;
    modelParameters.A = 1.0;
    modelParameters.externalForce = 0.0;

    // Valores por defecto
    algorithm = EULER_MARUYAMA;
    strcpy(algorithmText, "EULER_MARUYAMA");

    simulation.gamma = 1.0;
    simulation.dt = 0.001;
    simulation.totalTime = 1000.0;

    seed = 12345;
    useRandomSeed = true;

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
            else if (strcmp(name, "A") == 0)
            {
                modelParameters.A = atof(value);
            }
            else if (strcmp(name, "force") == 0)
            {
                modelParameters.externalForce = atof(value);
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
                if (strcmp(value, "random") == 0)
                {
                    useRandomSeed = true;
                }
                else
                {
                    seed = (unsigned int)strtoul(value, NULL, 10);
                    useRandomSeed = false;
                }
            }
            else if (strcmp(name, "x0") == 0)
            {
                simulation.initialPosition = atof(value);
            }
            else if (strcmp(name, "v0") == 0)
            {
                simulation.initialVelocity = atof(value);
            }
        }
    }

    fclose(configFile);

    seed = useRandomSeed ? generateRandomSeed() : seed;

    srand(seed);

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

    sprintf(
        outputFile,
        "data/raw/harmonic/harmonic_%s.csv",
        algorithmName);

    printf("\n");
    printf("Algorithm: %s\n", algorithmText);
    printf("eta = %g\n", simulation.gamma);
    printf("A = %g\n", modelParameters.A);
    printf("External force = %g\n", modelParameters.externalForce);
    printf("h = %g\n", simulation.dt);
    printf("Total time = %g\n", simulation.totalTime);
    printf("Seed = %u\n", seed);
    printf("\n");

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