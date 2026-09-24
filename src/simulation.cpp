#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>

void runSimulation(int model, int algorithm, SimulationParameters simulation, ModelParameters modelParameters,
                   char outputFile[])
{
    double x;
    double v;

    double time;
    double V;

    int numberOfSteps;
    int step;

    FILE *file;

    // Condiciones iniciales
    x = simulation.initialPosition;
    v = simulation.initialVelocity;

    // Número total de pasos
    numberOfSteps = (int)(simulation.totalTime / simulation.dt);

    // Abrimos el archivo de resultados
    file = fopen(outputFile, "w");

    // Comprobamos que se ha abierto correctamente
    if (file == NULL)
    {
        printf("Error opening output file\n");
        exit(1);
    }

    // Cabecera del CSV
    fprintf(file, "time,x,v,potential\n");

    // Bucle temporal
    for (step = 0; step <= numberOfSteps; step++)
    {
        // Tiempo actual
        time = step * simulation.dt;

        // Energía potencial actual
        V = potential(model, x, modelParameters);

        // Guardamos los resultados
        fprintf(file, "%lf,%lf,%lf,%lf\n", time, x, v, V);

        // Avanzamos un paso temporal
        if (step < numberOfSteps)
        {
            integrationStep(algorithm, model, &x, &v, simulation, modelParameters);
        }
    }

    // Cerramos el archivo
    fclose(file);
}