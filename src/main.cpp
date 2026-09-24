#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int model;
    int algorithm;

    SimulationParameters simulation;
    ModelParameters modelParameters;

    char algorithmName[30];
    char outputFile[100];

    // Inicializamos el generador de números aleatorios
    srand(time(NULL));

    // Elegimos el modelo físico
    model = HARMONIC;

    // Elegimos el algoritmo de integración
    algorithm = GJF;

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
    sprintf(outputFile, "data/raw/harmonic_%s.csv", algorithmName);

    // Parámetros temporales
    simulation.dt = 0.001;
    simulation.totalTime = 100.0;

    // Parámetros físicos
    simulation.mass = 1.0;
    simulation.gamma = 1.0;
    simulation.kBT = 1.0;

    // Condiciones iniciales
    simulation.initialPosition = 1.0;
    simulation.initialVelocity = 0.0;

    // Parámetro del oscilador armónico
    modelParameters.k = 1.0;

    // Ejecutamos la simulación
    runSimulation(
        model,
        algorithm,
        simulation,
        modelParameters,
        outputFile);

    // Mensaje al terminar
    printf("Simulation finished\n");
    printf("Results saved in: %s\n", outputFile);

    return 0;
}