#include "integrators.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void integrationStep(int algorithm, int model, double *x, double *v, SimulationParameters simulation,
                     ModelParameters modelParameters)
{

    if (algorithm == EULER_MARUYAMA)
    {
        eulerMaruyamaStep(model, x, v, simulation, modelParameters);
        return;
    }

    if (algorithm == RUNGE_KUTTA)
    {
        printf("Runge-Kutta not implemented yet\n");
        exit(1);
    }

    if (algorithm == GJF)
    {
        printf("G-JF not implemented yet\n");
        exit(1);
    }

    // Número de algoritmo incorrecto
    printf("Error: unknown integration algorithm\n");
    exit(1);
}

// Realiza un paso de Euler-Maruyama
void eulerMaruyamaStep(int model, double *x, double *v, SimulationParameters simulation,
                       ModelParameters modelParameters)
{
    double xOld;
    double vOld;
    double xNew;
    double vNew;
    double F;
    double Z;

    // Guardamos la posición y velocidad actuales
    xOld = *x;
    vOld = *v;

    // Calculamos la fuerza en la posición actual
    F = force(model, xOld, modelParameters);

    // Generamos el ruido gaussiano
    Z = randGaussian();

    // Actualizamos la posición
    xNew = xOld + vOld * simulation.dt;

    // Actualizamos la velocidad
    vNew = vOld + (F - simulation.gamma * vOld) * simulation.dt / simulation.mass + sqrt(2.0 * simulation.gamma * simulation.kBT * simulation.dt) * Z / simulation.mass;

    // Guardamos los nuevos valores
    *x = xNew;
    *v = vNew;
}

// Genera un número aleatorio gaussiano N(0,1) con Box-Muller
double randGaussian()
{
    static int saved = 0;
    static double z1;

    double u1;
    double u2;
    double radius;
    double z0;

    // Box-Muller genera dos gaussianos; reutilizamos el segundo
    if (saved == 1)
    {
        saved = 0;
        return z1;
    }

    // u1 debe ser mayor que 0 para evitar log(0)
    do
    {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 <= 0.0);

    u2 = (double)rand() / RAND_MAX;

    // Transformación de Box-Muller
    radius = sqrt(-2.0 * log(u1));
    z0 = radius * cos(2.0 * M_PI * u2);
    z1 = radius * sin(2.0 * M_PI * u2);

    saved = 1;

    return z0;
}