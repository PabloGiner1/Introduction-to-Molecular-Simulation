#include "simulation.hpp"
<<<<<<< HEAD
#include "integrators.hpp"

void run_simulation(const Parameters& parameters) {
    switch (parameters.algorithm) {
        case Algorithm::EulerMaruyama:
            Euler_Maruyama(
                parameters.x0,
                parameters.p0,
                parameters.eta,
                parameters.h,
                parameters.t_max
            );
            break;

        case Algorithm::Runge_Kutta:
            // Runge-Kutta y Verlet estocastico...

            break;
    }
=======

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

    // Comprobamos que se ha abierto correctamente
    if (file == NULL)
    {
        printf("Error opening output file\n");
        exit(1);
    }

    // Guardamos los parámetros usados en la simulación
    parametersFile = fopen("data/raw/parameters.txt", "w");

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
        // Tiempo actual
        time = step * simulation.dt;

        // Guardamos solo algunas medidas para no crear archivos enormes
        if (step % saveEvery == 0 || step == numberOfSteps)
        {
            // Energía cinética y potencial
            kineticEnergy = 0.5 * simulation.mass * v * v;
            potentialEnergy = potential(model, x, modelParameters);

            // Guardamos los resultados
            fprintf(file, "%lf,%lf,%lf,%lf,%lf\n",
                    time, x, v, kineticEnergy, potentialEnergy);
        }

        // Avanzamos un paso temporal
        if (step < numberOfSteps)
        {
            integrationStep(algorithm, model, &x, &v, simulation, modelParameters);
        }
    }

    // Cerramos el archivo
    fclose(file);
>>>>>>> 6e59527563090f56cb92d37dd26ef7dfaeef89bd
}