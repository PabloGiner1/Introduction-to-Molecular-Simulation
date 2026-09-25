<<<<<<< HEAD
#pragma once

double rand_gaussian();

void guardar_parametros(
    double x,
    double p,
    double eta,
    double h,
    double t_max
);

void Euler_Maruyama(
    double x,
    double p,
    double eta,
    double h,
    double t_max
);
=======
#ifndef INTEGRATORS_HPP
#define INTEGRATORS_HPP

#include "models.hpp"

// Identificadores de los algoritmos
#define EULER_MARUYAMA 1
#define RUNGE_KUTTA 2
#define GJF 3

// Parámetros generales de la simulación
struct SimulationParameters
{
    double dt;        // Paso temporal
    double totalTime; // Tiempo total

    double mass;  // Masa
    double gamma; // Rozamiento
    double kBT;   // Energía térmica kB*T

    double initialPosition; // Posición inicial
    double initialVelocity; // Velocidad inicial
};

// Selecciona el algoritmo y realiza un paso
void integrationStep(int algorithm, int model, double *x, double *v, SimulationParameters simulation,
                     ModelParameters modelParameters);

// Los que hay que hacer para que realice un paso de cada algoritmo
void eulerMaruyamaStep(int model, double *x, double *v, SimulationParameters simulation,
                       ModelParameters modelParameters);

void rungeKuttaStep(int model, double *x, double *v, SimulationParameters simulation,
                    ModelParameters modelParameters);

void gjfStep(int model, double *x, double *v, SimulationParameters simulation,
             ModelParameters modelParameters);

// Genera un número aleatorio gaussiano N(0,1)
double randGaussian();

#endif
>>>>>>> 6e59527563090f56cb92d37dd26ef7dfaeef89bd
