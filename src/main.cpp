#include "simulation.hpp"
#include "polymer.hpp"

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
    PolymerParameters polymerParameters;

    char algorithmText[50];
    char algorithmName[30];

    char outputFile[150];

    char line[200];
    char name[50];
    char value[100];

    FILE *configFile;

    // Modelo por defecto
    model = DOUBLE_WELL;

    // Parámetros generales
    simulation.mass = 1.0;
    simulation.kBT = 0.2;

    simulation.initialPosition = 1.0;
    simulation.initialVelocity = 0.0;

    simulation.maxSavedPoints = 10000;
    simulation.minThermalizationTime = 0.0;

    // Parámetros de los modelos escalares
    modelParameters.k = 1.0;
    modelParameters.A = 1.0;
    modelParameters.externalForce = 0.0;

    // Parámetros del polímero
    polymerParameters.N = 8;
    polymerParameters.springConstant = 100.0;
    polymerParameters.equilibriumLength = 1.0;

    // Valores por defecto
    algorithm = EULER_MARUYAMA;

    strcpy(
        algorithmText,
        "EULER_MARUYAMA");

    simulation.gamma = 1.0;
    simulation.dt = 0.001;
    simulation.totalTime = 1000.0;

    seed = 12345;
    useRandomSeed = true;

    configFile = fopen(
        "config/simulation.txt",
        "r");

    if (configFile == NULL)
    {
        printf(
            "Error opening config/simulation.txt\n");

        return 1;
    }

    while (
        fgets(
            line,
            sizeof(line),
            configFile) != NULL)
    {
        if (
            line[0] == '#' || line[0] == '\n')
        {
            continue;
        }

        if (
            sscanf(
                line,
                " %49[^=]=%99s",
                name,
                value) == 2)
        {
            if (
                strcmp(
                    name,
                    "model") == 0)
            {
                if (
                    strcmp(
                        value,
                        "HARMONIC") == 0)
                {
                    model = HARMONIC;
                }

                else if (
                    strcmp(
                        value,
                        "DOUBLE_WELL") == 0)
                {
                    model = DOUBLE_WELL;
                }

                else if (
                    strcmp(
                        value,
                        "POLYMER") == 0)
                {
                    model = POLYMER;
                }

                else
                {
                    printf(
                        "Error: unknown model\n");

                    fclose(
                        configFile);

                    return 1;
                }
            }

            else if (
                strcmp(
                    name,
                    "algorithm") == 0)
            {
                strcpy(
                    algorithmText,
                    value);

                if (
                    strcmp(
                        value,
                        "EULER_MARUYAMA") == 0)
                {
                    algorithm =
                        EULER_MARUYAMA;
                }

                else if (
                    strcmp(
                        value,
                        "RUNGE_KUTTA") == 0)
                {
                    algorithm =
                        RUNGE_KUTTA;
                }

                else if (
                    strcmp(
                        value,
                        "GJF") == 0)
                {
                    algorithm =
                        GJF;
                }

                else
                {
                    printf(
                        "Error: unknown algorithm\n");

                    fclose(
                        configFile);

                    return 1;
                }
            }

            else if (
                strcmp(
                    name,
                    "eta") == 0)
            {
                simulation.gamma =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "kBT") == 0)
            {
                simulation.kBT =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "mass") == 0)
            {
                simulation.mass =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "A") == 0)
            {
                modelParameters.A =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "force") == 0)
            {
                modelParameters.externalForce =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "N") == 0)
            {
                polymerParameters.N =
                    atoi(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "ke") == 0)
            {
                polymerParameters.springConstant =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "l0") == 0)
            {
                polymerParameters.equilibriumLength =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "h") == 0)
            {
                simulation.dt =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "totalTime") == 0)
            {
                simulation.totalTime =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "maxSavedPoints") == 0)
            {
                simulation.maxSavedPoints =
                    atoi(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "minThermalizationTime") == 0)
            {
                simulation.minThermalizationTime =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "seed") == 0)
            {
                if (
                    strcmp(
                        value,
                        "random") == 0)
                {
                    useRandomSeed = true;
                }

                else
                {
                    seed =
                        (unsigned int)
                            strtoul(
                                value,
                                NULL,
                                10);

                    useRandomSeed = false;
                }
            }

            else if (
                strcmp(
                    name,
                    "x0") == 0)
            {
                simulation.initialPosition =
                    atof(
                        value);
            }

            else if (
                strcmp(
                    name,
                    "v0") == 0)
            {
                simulation.initialVelocity =
                    atof(
                        value);
            }
        }
    }

    fclose(
        configFile);

    if (
        simulation.maxSavedPoints < 1)
    {
        simulation.maxSavedPoints = 10000;
    }

    if (
        simulation.minThermalizationTime < 0.0)
    {
        simulation.minThermalizationTime = 0.0;
    }

    seed =
        useRandomSeed
            ? generateRandomSeed()
            : seed;

    srand(
        seed);

    if (
        algorithm == EULER_MARUYAMA)
    {
        strcpy(
            algorithmName,
            "euler");
    }

    else if (
        algorithm == RUNGE_KUTTA)
    {
        strcpy(
            algorithmName,
            "rk");
    }

    else
    {
        strcpy(
            algorithmName,
            "gjf");
    }

    printf("\n");

    printf(
        "Algorithm: %s\n",
        algorithmText);

    printf(
        "eta = %g\n",
        simulation.gamma);

    printf(
        "kBT = %g\n",
        simulation.kBT);

    printf(
        "h = %g\n",
        simulation.dt);

    printf(
        "Total time = %g\n",
        simulation.totalTime);

    printf(
        "Minimum thermalization time = %g\n",
        simulation.minThermalizationTime);

    printf(
        "Max saved points = %d\n",
        simulation.maxSavedPoints);

    printf(
        "Seed = %u\n",
        seed);

    printf("\n");

    // Polímero
    if (
        model == POLYMER)
    {
        printf(
            "Model: POLYMER\n");

        printf(
            "N = %d\n",
            polymerParameters.N);

        printf(
            "ke = %g\n",
            polymerParameters.springConstant);

        printf(
            "l0 = %g\n",
            polymerParameters.equilibriumLength);

        printf("\n");

        strcpy(
            outputFile,
            "data/raw/polymer/polymer.csv");

        runPolymerSimulation(
            algorithm,
            simulation,
            polymerParameters,
            outputFile);

        return 0;
    }

    if (
        model == HARMONIC)
    {
        printf(
            "Model: HARMONIC\n");
    }

    if (
        model == DOUBLE_WELL)
    {
        printf(
            "Model: DOUBLE_WELL\n");

        printf(
            "A = %g\n",
            modelParameters.A);

        printf(
            "External force = %g\n",
            modelParameters.externalForce);
    }

    printf("\n");

    sprintf(
        outputFile,
        "data/raw/harmonic/harmonic_%s.csv",
        algorithmName);

    runSimulation(
        model,
        algorithm,
        simulation,
        modelParameters,
        outputFile);

    printf(
        "Simulation finished\n");

    printf(
        "Results saved in: %s\n",
        outputFile);

    return 0;
}