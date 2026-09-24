#include "models.hpp"

#include <stdio.h>
#include <stdlib.h>

double force(int model, double x, ModelParameters parameters)
{
    // Oscilador armónico: F = -k*x
    if (model == HARMONIC)
    {
        return -parameters.k * x;
    }

    // Hay que completar estos dos aun :)
    if (model == DOUBLE_WELL)
    {
        printf("Error: model not implemented\n");
    }

    if (model == POLYMER)
    {
        printf("Error: model not implemented\n");
    }

    exit(1);
}

double potential(int model, double x, ModelParameters parameters)
{
    // Oscilador armónico: V = 1/2*k*x^2
    if (model == HARMONIC)
    {
        return 0.5 * parameters.k * x * x;
    }

    // Hay que completar estos dos aun :)
    if (model == DOUBLE_WELL)
    {
        printf("Error: model not implemented\n");
    }

    if (model == POLYMER)
    {
        printf("Error: model not implemented\n");
    }

    exit(1);
}