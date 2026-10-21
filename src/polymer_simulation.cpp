#include "polymer.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define POLYMER_WINDOW 5000

void runPolymerSimulation(
    int algorithm,
    SimulationParameters simulation,
    PolymerParameters polymer,
    char outputFile[])
{
    double x[MAX_MONOMERS];
    double y[MAX_MONOMERS];
    double z[MAX_MONOMERS];

    double vx[MAX_MONOMERS];
    double vy[MAX_MONOMERS];
    double vz[MAX_MONOMERS];

    double kineticEnergy;
    double elasticEnergy;

    double velocitySum;
    double velocitySquaredSum;

    double velocityMean;
    double velocityVariance;
    double velocityStd;

    double theoreticalVelocityStd;

    double tolerance;

    double time;

    int numberOfSteps;
    int maxSavedPoints;
    int saveEvery;

    int thermalizationStep;
    int windowStep;

    int consecutiveWindows;
    int requiredWindows;

    int thermalized;

    FILE *file;
    FILE *parametersFile;

    if (algorithm != GJF)
    {
        printf(
            "Error: the first polymer implementation "
            "only supports G-JF\n");

        exit(1);
    }

    if (
        polymer.N < 2 || polymer.N > MAX_MONOMERS)
    {
        printf(
            "Error: invalid number of monomers\n");

        exit(1);
    }

    initializePolymer(
        x,
        y,
        z,
        vx,
        vy,
        vz,
        polymer);

    numberOfSteps =
        (int)(simulation.totalTime / simulation.dt);

    maxSavedPoints = 5000;

    saveEvery =
        numberOfSteps / maxSavedPoints;

    if (saveEvery < 1)
    {
        saveEvery = 1;
    }

    theoreticalVelocityStd = sqrt(
        simulation.kBT / simulation.mass);

    tolerance = 0.05;

    requiredWindows = 3;
    consecutiveWindows = 0;

    thermalized = 0;

    thermalizationStep = 0;

    printf(
        "Thermalizing polymer...\n");

    while (thermalized == 0)
    {
        velocitySum = 0.0;
        velocitySquaredSum = 0.0;

        for (
            windowStep = 0;
            windowStep < POLYMER_WINDOW;
            windowStep++)
        {
            polymerGJFStep(
                x,
                y,
                z,
                vx,
                vy,
                vz,
                simulation,
                polymer);

            thermalizationStep++;

            for (int i = 0; i < polymer.N; i++)
            {
                velocitySum += vx[i];
                velocitySum += vy[i];
                velocitySum += vz[i];

                velocitySquaredSum +=
                    vx[i] * vx[i];

                velocitySquaredSum +=
                    vy[i] * vy[i];

                velocitySquaredSum +=
                    vz[i] * vz[i];
            }
        }

        int samples =
            POLYMER_WINDOW * polymer.N * 3;

        velocityMean =
            velocitySum / samples;

        velocityVariance =
            velocitySquaredSum / samples - velocityMean * velocityMean;

        if (velocityVariance < 0.0)
        {
            velocityVariance = 0.0;
        }

        velocityStd =
            sqrt(
                velocityVariance);

        printf(
            "Thermalization step %d: "
            "sigma_v = %.5f "
            "(theory %.5f)\n",
            thermalizationStep,
            velocityStd,
            theoreticalVelocityStd);

        if (
            fabs(
                (
                    velocityStd - theoreticalVelocityStd) /
                theoreticalVelocityStd) < tolerance)
        {
            consecutiveWindows++;
        }
        else
        {
            consecutiveWindows = 0;
        }

        if (
            consecutiveWindows >= requiredWindows)
        {
            thermalized = 1;
        }

        if (
            thermalizationStep >= numberOfSteps)
        {
            printf(
                "Error: polymer thermalization "
                "was not reached\n");

            exit(1);
        }
    }

    printf(
        "Polymer thermalized after %d steps "
        "(t = %.3f)\n",
        thermalizationStep,
        thermalizationStep * simulation.dt);

    file = fopen(
        outputFile,
        "w");

    if (file == NULL)
    {
        printf(
            "Error opening polymer output file\n");

        exit(1);
    }

    // Cabecera
    fprintf(
        file,
        "time,kinetic,elastic");

    for (int i = 0; i < polymer.N; i++)
    {
        fprintf(
            file,
            ",x%d,y%d,z%d",
            i,
            i,
            i);
    }

    fprintf(
        file,
        "\n");

    // Producción
    for (
        int step = 0;
        step <= numberOfSteps;
        step++)
    {
        time =
            step * simulation.dt;

        if (
            step % saveEvery == 0 || step == numberOfSteps)
        {
            kineticEnergy =
                polymerKineticEnergy(
                    vx,
                    vy,
                    vz,
                    simulation,
                    polymer);

            elasticEnergy =
                polymerElasticEnergy(
                    x,
                    y,
                    z,
                    polymer);

            fprintf(
                file,
                "%.10f,%.10f,%.10f",
                time,
                kineticEnergy,
                elasticEnergy);

            for (int i = 0; i < polymer.N; i++)
            {
                fprintf(
                    file,
                    ",%.10f,%.10f,%.10f",
                    x[i],
                    y[i],
                    z[i]);
            }

            fprintf(
                file,
                "\n");
        }

        if (step < numberOfSteps)
        {
            polymerGJFStep(
                x,
                y,
                z,
                vx,
                vy,
                vz,
                simulation,
                polymer);
        }
    }

    fclose(file);

    parametersFile = fopen(
        "data/raw/polymer/parameters.txt",
        "w");

    if (parametersFile != NULL)
    {
        fprintf(
            parametersFile,
            "N=%d\n",
            polymer.N);

        fprintf(
            parametersFile,
            "ke=%lf\n",
            polymer.springConstant);

        fprintf(
            parametersFile,
            "l0=%lf\n",
            polymer.equilibriumLength);

        fprintf(
            parametersFile,
            "mass=%lf\n",
            simulation.mass);

        fprintf(
            parametersFile,
            "gamma=%lf\n",
            simulation.gamma);

        fprintf(
            parametersFile,
            "kBT=%lf\n",
            simulation.kBT);

        fprintf(
            parametersFile,
            "dt=%lf\n",
            simulation.dt);

        fprintf(
            parametersFile,
            "totalTime=%lf\n",
            simulation.totalTime);

        fprintf(
            parametersFile,
            "algorithm=%d\n",
            algorithm);

        fprintf(
            parametersFile,
            "thermalizationSteps=%d\n",
            thermalizationStep);

        fprintf(
            parametersFile,
            "thermalizationTime=%lf\n",
            thermalizationStep * simulation.dt);

        fprintf(
            parametersFile,
            "saveEvery=%d\n",
            saveEvery);

        fprintf(
            parametersFile,
            "savedDt=%lf\n",
            saveEvery * simulation.dt);

        fclose(
            parametersFile);
    }

    printf(
        "Polymer simulation finished\n");

    printf(
        "Results saved in: %s\n",
        outputFile);
}