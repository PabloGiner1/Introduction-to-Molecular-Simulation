<<<<<<< HEAD
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "integrators.hpp"
#include"simulation.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

int main(void) {

    // Empezamos a poner los parametros necesarios para Euler-Maruyama en el OA
    
    std::cout << "TFIII - Simulacion Molecular\n";

    Parameters parametros;
 
    parametros.x0 = 1.0;                                            // Posición inicial
    parametros.p0 = 0.0;                                            // Momento inicial
    parametros.eta = 10.0;                                           // Amortiguamiento: 0.1, 1.0, 10
    parametros.h = 0.0001;                                          // Paso de tiempo: 0.1, 0.01, 0.001, 0.0001
    parametros.t_max = 10000.0;                                     // Tiempo máximo de simulación
    parametros.algorithm = Algorithm::EulerMaruyama;                //Alrgoritmo de integración: Euler-Maruyama, Runge-Kutta, Verlet



    rand_gaussian(); // Llamada inicial para establecer la semilla
    guardar_parametros(
        parametros.x0,
        parametros.p0,
        parametros.eta,
        parametros.h,
        parametros.t_max
    );
    printf("Parametros guardados en data/raw/parameters.txt\n");

    run_simulation(parametros);
    std::cout << "Compilacion y ejecucion correctas.\n";
=======
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
    algorithm = EULER_MARUYAMA;

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
>>>>>>> 6e59527563090f56cb92d37dd26ef7dfaeef89bd

    return 0;
}