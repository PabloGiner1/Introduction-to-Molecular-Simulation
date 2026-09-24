#ifndef MODELS_H
#define MODELS_H

// Identificadores de los modelos físicos
#define HARMONIC 1
#define DOUBLE_WELL 2
#define POLYMER 3

// Parámetros propios del modelo
struct ModelParameters
{
    double k; // Constante elástica del oscilador
};

// Calcula la fuerza en la posición x
double force(int model, double x, ModelParameters parameters);

// Calcula la energía potencial en x
double potential(int model, double x, ModelParameters parameters);

#endif