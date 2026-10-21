#include "polymer.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void initializePolymer(
    double x[],
    double y[],
    double z[],
    double vx[],
    double vy[],
    double vz[],
    PolymerParameters polymer)
{
    for (int i = 0; i < polymer.N; i++)
    {
        // Cadena recta a lo largo del eje x
        x[i] = i * polymer.equilibriumLength;
        y[i] = 0.0;
        z[i] = 0.0;

        vx[i] = 0.0;
        vy[i] = 0.0;
        vz[i] = 0.0;
    }
}

void polymerForces(
    double x[],
    double y[],
    double z[],
    double fx[],
    double fy[],
    double fz[],
    PolymerParameters polymer)
{
    double dx;
    double dy;
    double dz;

    double distance;
    double forceMagnitude;

    double forceX;
    double forceY;
    double forceZ;

    // Inicializamos todas las fuerzas a cero
    for (int i = 0; i < polymer.N; i++)
    {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    // Fuerzas entre monómeros consecutivos
    for (int i = 0; i < polymer.N - 1; i++)
    {
        dx = x[i + 1] - x[i];
        dy = y[i + 1] - y[i];
        dz = z[i + 1] - z[i];

        distance = sqrt(
            dx * dx + dy * dy + dz * dz);

        if (distance < 1.0e-12)
        {
            continue;
        }

        forceMagnitude =
            polymer.springConstant * (distance - polymer.equilibriumLength);

        forceX =
            forceMagnitude * dx / distance;

        forceY =
            forceMagnitude * dy / distance;

        forceZ =
            forceMagnitude * dz / distance;

        // Acción sobre el monómero i
        fx[i] += forceX;
        fy[i] += forceY;
        fz[i] += forceZ;

        // Reacción sobre el monómero i+1
        fx[i + 1] -= forceX;
        fy[i + 1] -= forceY;
        fz[i + 1] -= forceZ;
    }
}

double polymerKineticEnergy(
    double vx[],
    double vy[],
    double vz[],
    SimulationParameters simulation,
    PolymerParameters polymer)
{
    double energy = 0.0;

    for (int i = 0; i < polymer.N; i++)
    {
        energy +=
            0.5 * simulation.mass * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    return energy;
}

double polymerElasticEnergy(
    double x[],
    double y[],
    double z[],
    PolymerParameters polymer)
{
    double energy = 0.0;

    double dx;
    double dy;
    double dz;

    double distance;
    double extension;

    for (int i = 0; i < polymer.N - 1; i++)
    {
        dx = x[i + 1] - x[i];
        dy = y[i + 1] - y[i];
        dz = z[i + 1] - z[i];

        distance = sqrt(
            dx * dx + dy * dy + dz * dz);

        extension =
            distance - polymer.equilibriumLength;

        energy +=
            0.5 * polymer.springConstant * extension * extension;
    }

    return energy;
}

void polymerGJFStep(
    double x[],
    double y[],
    double z[],
    double vx[],
    double vy[],
    double vz[],
    SimulationParameters simulation,
    PolymerParameters polymer)
{
    double fx[MAX_MONOMERS];
    double fy[MAX_MONOMERS];
    double fz[MAX_MONOMERS];

    double fxNew[MAX_MONOMERS];
    double fyNew[MAX_MONOMERS];
    double fzNew[MAX_MONOMERS];

    double xNew[MAX_MONOMERS];
    double yNew[MAX_MONOMERS];
    double zNew[MAX_MONOMERS];

    double vxNew[MAX_MONOMERS];
    double vyNew[MAX_MONOMERS];
    double vzNew[MAX_MONOMERS];

    double betaX[MAX_MONOMERS];
    double betaY[MAX_MONOMERS];
    double betaZ[MAX_MONOMERS];

    double alpha;
    double a;
    double b;

    double noiseAmplitude;

    alpha =
        simulation.gamma * simulation.dt / (2.0 * simulation.mass);

    a =
        (1.0 - alpha) / (1.0 + alpha);

    b =
        1.0 / (1.0 + alpha);

    noiseAmplitude = sqrt(
        2.0 * simulation.gamma * simulation.mass * simulation.kBT * simulation.dt);

    // Fuerzas en las posiciones actuales
    polymerForces(
        x,
        y,
        z,
        fx,
        fy,
        fz,
        polymer);

    // Nuevas posiciones
    for (int i = 0; i < polymer.N; i++)
    {
        betaX[i] =
            noiseAmplitude * randGaussian();

        betaY[i] =
            noiseAmplitude * randGaussian();

        betaZ[i] =
            noiseAmplitude * randGaussian();

        xNew[i] =
            x[i] + b * simulation.dt * vx[i] + b * simulation.dt * simulation.dt * fx[i] / (2.0 * simulation.mass) + b * simulation.dt * betaX[i] / (2.0 * simulation.mass);

        yNew[i] =
            y[i] + b * simulation.dt * vy[i] + b * simulation.dt * simulation.dt * fy[i] / (2.0 * simulation.mass) + b * simulation.dt * betaY[i] / (2.0 * simulation.mass);

        zNew[i] =
            z[i] + b * simulation.dt * vz[i] + b * simulation.dt * simulation.dt * fz[i] / (2.0 * simulation.mass) + b * simulation.dt * betaZ[i] / (2.0 * simulation.mass);
    }

    // Fuerzas en las nuevas posiciones
    polymerForces(
        xNew,
        yNew,
        zNew,
        fxNew,
        fyNew,
        fzNew,
        polymer);

    // Nuevas velocidades
    for (int i = 0; i < polymer.N; i++)
    {
        vxNew[i] =
            a * vx[i] + simulation.dt * (a * fx[i] + fxNew[i]) / (2.0 * simulation.mass) + b * betaX[i] / simulation.mass;

        vyNew[i] =
            a * vy[i] + simulation.dt * (a * fy[i] + fyNew[i]) / (2.0 * simulation.mass) + b * betaY[i] / simulation.mass;

        vzNew[i] =
            a * vz[i] + simulation.dt * (a * fz[i] + fzNew[i]) / (2.0 * simulation.mass) + b * betaZ[i] / simulation.mass;

        x[i] = xNew[i];
        y[i] = yNew[i];
        z[i] = zNew[i];

        vx[i] = vxNew[i];
        vy[i] = vyNew[i];
        vz[i] = vzNew[i];
    }
}