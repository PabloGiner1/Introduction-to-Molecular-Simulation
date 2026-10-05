#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>

void runSimulation(int model, int algorithm, SimulationParameters simulation, ModelParameters modelParameters,
                   char outputFile[])
{
    double x;
    double v;

    double time;
    double kineticEnergy;
    double potentialEnergy;

    int numberOfSteps;
    int maxSavedPoints;
    int saveEvery;
    int step;

    FILE *file;
    FILE *parametersFile;

    // Condiciones iniciales
    x = simulation.initialPosition;
    v = simulation.initialVelocity;

    // Número total de pasos
    numberOfSteps = (int)(simulation.totalTime / simulation.dt);

    // Guardamos como máximo unas 10000 medidas
    maxSavedPoints = 10000;
    saveEvery = numberOfSteps / maxSavedPoints;

    if (saveEvery < 1)
    {
        saveEvery = 1;
    }

    // Abrimos el archivo de resultados
    file = fopen(outputFile, "w");

    if (file == NULL)
    {
        printf("Error opening output file\n");
        exit(1);
    }

    // Guardamos los parámetros usados
    parametersFile = fopen(
        "data/raw/harmonic/parameters.txt",
        "w");

    if (parametersFile != NULL)
    {
        fprintf(parametersFile, "x0=%lf\n", simulation.initialPosition);
        fprintf(parametersFile, "v0=%lf\n", simulation.initialVelocity);
        fprintf(parametersFile, "gamma=%lf\n", simulation.gamma);
        fprintf(parametersFile, "dt=%lf\n", simulation.dt);
        fprintf(parametersFile, "totalTime=%lf\n", simulation.totalTime);
        fprintf(parametersFile, "mass=%lf\n", simulation.mass);
        fprintf(parametersFile, "kBT=%lf\n", simulation.kBT);
        fprintf(parametersFile, "k=%lf\n", modelParameters.k);
        fprintf(parametersFile, "algorithm=%d\n", algorithm);

        fclose(parametersFile);
    }

    // Cabecera del CSV
    fprintf(file, "time,x,v,kinetic,potential\n");

    // Bucle temporal
    for (step = 0; step <= numberOfSteps; step++)
    {
        time = step * simulation.dt;

        if (step % saveEvery == 0 || step == numberOfSteps)
        {
            kineticEnergy = 0.5 * simulation.mass * v * v;
            potentialEnergy = potential(model, x, modelParameters);

            fprintf(
                file,
                "%lf,%lf,%lf,%lf,%lf\n",
                time,
                x,
                v,
                kineticEnergy,
                potentialEnergy);
        }

        if (step < numberOfSteps)
        {
            integrationStep(
                algorithm,
                model,
                &x,
                &v,
                simulation,
                modelParameters);
        }
    }

    fclose(file);
}