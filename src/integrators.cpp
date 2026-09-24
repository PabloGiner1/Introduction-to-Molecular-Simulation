#include "integrators.hpp"

#include <stdio.h>
#include <stdlib.h>

void integrationStep(int algorithm, int model, double *x, double *v, SimulationParameters simulation,
                     ModelParameters modelParameters)
{

    if (algorithm == EULER_MARUYAMA)
    {
        printf("Euler-Maruyama not implemented yet\n");
        exit(1);
    }

    if (algorithm == RUNGE_KUTTA)
    {
        printf("Runge-Kutta not implemented yet\n");
        exit(1);
    }

    // Número de algoritmo incorrecto
    printf("Error: unknown integration algorithm\n");
    exit(1);
}