#include "models.hpp"

#include <stdio.h>
#include <stdlib.h>

double force(int model, double x, ModelParameters parameters)
{
    // Oscilador armónico
    if (model == HARMONIC)
    {
        return -parameters.k * x;
    }

    // Doble pozo con fuerza externa constante
    if (model == DOUBLE_WELL)
    {
        return -4.0 * parameters.A * x * (x * x - 1.0) + parameters.externalForce;
    }

    if (model == POLYMER)
    {
        printf("Error: model not implemented\n");
    }

    exit(1);
}

double potential(int model, double x, ModelParameters parameters)
{
    // Oscilador armónico
    if (model == HARMONIC)
    {
        return 0.5 * parameters.k * x * x;
    }

    // V(x) = A(x^2 - 1)^2 - F*x
    if (model == DOUBLE_WELL)
    {
        return parameters.A * (x * x - 1.0) * (x * x - 1.0) - parameters.externalForce * x;
    }

    if (model == POLYMER)
    {
        printf("Error: model not implemented\n");
    }

    exit(1);
}