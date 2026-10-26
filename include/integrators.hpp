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
    double dt;
    double totalTime;

    double mass;
    double gamma;
    double kBT;

    double initialPosition;
    double initialVelocity;

    int maxSavedPoints;

    // Tiempo mínimo que debe pasar
    // antes de aceptar la termalización.
    double minThermalizationTime;
};

// Selecciona el algoritmo y realiza un paso
void integrationStep(
    int algorithm,
    int model,
    double *x,
    double *v,
    SimulationParameters simulation,
    ModelParameters modelParameters);

// Euler-Maruyama
void eulerMaruyamaStep(
    int model,
    double *x,
    double *v,
    SimulationParameters simulation,
    ModelParameters modelParameters);

// Runge-Kutta estocástico
void rungeKuttaStep(
    int model,
    double *x,
    double *v,
    SimulationParameters simulation,
    ModelParameters modelParameters);

// Grønbech-Jensen
void gjfStep(
    int model,
    double *x,
    double *v,
    SimulationParameters simulation,
    ModelParameters modelParameters);

// Genera N(0,1)
double randGaussian();

// Genera una semilla aleatoria
unsigned int generateRandomSeed();

#endif