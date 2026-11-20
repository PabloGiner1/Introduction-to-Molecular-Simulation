#ifndef POLYMER_HPP
#define POLYMER_HPP

#include "integrators.hpp"

#define MAX_MONOMERS 100

struct PolymerParameters
{
    int N;
    double springConstant;
    double equilibriumLength;
    double externalForce;
    bool firstEndFixed;
};

// Inicializa una cadena recta
void initializePolymer(
    double x[],
    double y[],
    double z[],
    double vx[],
    double vy[],
    double vz[],
    PolymerParameters polymer);

// Calcula las fuerzas elásticas
void polymerForces(
    double x[],
    double y[],
    double z[],
    double fx[],
    double fy[],
    double fz[],
    PolymerParameters polymer);

// Energía cinética total
double polymerKineticEnergy(
    double vx[],
    double vy[],
    double vz[],
    SimulationParameters simulation,
    PolymerParameters polymer);

// Energía elástica total
double polymerElasticEnergy(
    double x[],
    double y[],
    double z[],
    PolymerParameters polymer);

// Un paso G-JF del polímero
void polymerGJFStep(
    double x[],
    double y[],
    double z[],
    double vx[],
    double vy[],
    double vz[],
    SimulationParameters simulation,
    PolymerParameters polymer);

// Simulación completa
void runPolymerSimulation(
    int algorithm,
    SimulationParameters simulation,
    PolymerParameters polymer,
    char outputFile[]);

// Calcula la posición del centro de masas (cmX, cmY, cmZ)
void polymerCenterOfMass(
    double x[],
    double y[],
    double z[],
    double *cmX,
    double *cmY,
    double *cmZ,
    PolymerParameters polymer);

// Calcula el radio de giro al cuadrado (Rg^2) y el radio de giro (Rg)
double polymerRadiusOfGyrationSquared(
    double x[],
    double y[],
    double z[],
    PolymerParameters polymer);

double polymerRadiusOfGyration(
    double x[],
    double y[],
    double z[],
    PolymerParameters polymer);

// Calcula la distancia de extremo a extremo (Ree)
double polymerEndToEndDistance(
    double x[],
    double y[],
    double z[],
    PolymerParameters polymer);

#endif