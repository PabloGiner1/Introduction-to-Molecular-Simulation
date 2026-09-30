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
        rungeKuttaStochasticStep(model, x, v, simulation, modelParameters);
        return;
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

void rungeKuttaStochasticStep(int model, double *x, double *v, SimulationParameters simulation, ModelParameters modelParameters)
{
    double xOld = *x;
    double vOld = *v;
    double Z;
    double F1, G1, F2, G2;
    double pOld, pPred;
    
    // Generamos el número aleatorio gaussiano puro N(0,1)
    double zeta = randGaussian();
    
    // Calculamos el factor de ruido Z (escalado con las constantes físicas)
    // Z = sqrt(2 * gamma * m * kBT * dt) * zeta
    Z = sqrt(2.0 * simulation.gamma * simulation.mass * simulation.kBT * simulation.dt) * zeta;
    
    // Para simplificar la fracción, calculamos el momento lineal actual
    pOld = vOld * simulation.mass;

    // --- 1ª ETAPA ---
    F1 = (pOld + Z) / simulation.mass; 
    G1 = force(model, xOld, modelParameters) - (simulation.gamma * (pOld + Z));

    // --- 2ª ETAPA ---
    // Predecimos la posición y el momento en el futuro (dt)
    double xPred = xOld + (simulation.dt * F1);
    pPred = pOld + (simulation.dt * G1);
    
    F2 = pPred / simulation.mass;
    G2 = force(model, xPred, modelParameters) - (simulation.gamma * pPred);

    // --- ACTUALIZACIÓN FINAL ---
    // Sobrescribimos directamente en los punteros la media de las dos etapas
    *x = xOld + 0.5 * simulation.dt * (F1 + F2);
    
    // Calculamos el nuevo momento y lo dividimos entre la masa para guardar la velocidad
    *v = (pOld + 0.5 * simulation.dt * (G1 + G2) + Z) / simulation.mass;
}