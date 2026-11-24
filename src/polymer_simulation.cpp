#include "polymer.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define POLYMER_WINDOW 200000
#define POLYMER_MAX_THERMALIZATION_TIME 100000.0

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

    double elasticEnergySum;
    double elasticEnergySquaredSum;
    double elasticEnergyMean;
    double elasticEnergyVariance;
    double elasticEnergyStd;
    double previousElasticEnergyMean;
    double previousElasticEnergyStd;

    double tolerance;

    double time;

    int numberOfSteps;

    int maxSavedPoints;
    int saveEvery;

    int minThermalizationSteps;
    int maxThermalizationSteps;

    int thermalizationStep;
    int windowStep;
    int stepsInWindow;

    int consecutiveWindows;
    int requiredWindows;

    int hasPreviousWindow;
    int thermalized;

    FILE *file;
    FILE *parametersFile;

    if (
        algorithm != GJF)
    {
        printf(
            "Error: polymer simulation "
            "only supports G-JF\n");

        exit(1);
    }

    if (
        polymer.N < 2 || polymer.N > MAX_MONOMERS)
    {
        printf(
            "Error: invalid number "
            "of monomers\n");

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

    maxSavedPoints =
        simulation.maxSavedPoints;

    if (
        maxSavedPoints < 1)
    {
        maxSavedPoints = 5000;
    }

    saveEvery =
        numberOfSteps / maxSavedPoints;

    if (
        saveEvery < 1)
    {
        saveEvery = 1;
    }

    minThermalizationSteps =
        (int)ceil(
            simulation.minThermalizationTime / simulation.dt);

    maxThermalizationSteps =
        (int)(
            POLYMER_MAX_THERMALIZATION_TIME /
            simulation.dt);

    // if (
    //     minThermalizationSteps > numberOfSteps)
    // {
    //     printf(
    //         "Error: minimum thermalization "
    //         "time is larger than totalTime\n");

    //     exit(1);
    // }

    tolerance = 0.05;

    requiredWindows = 10;

    consecutiveWindows = 0;
    hasPreviousWindow = 0;

    thermalized = 0;

    thermalizationStep = 0;

    printf(
        "Thermalizing polymer...\n");

    printf(
        "N = %d\n",
        polymer.N);

    printf(
        "Minimum thermalization time = %g\n",
        simulation.minThermalizationTime);

    while (
        thermalized == 0 &&
        thermalizationStep < maxThermalizationSteps)
    {
        elasticEnergySum = 0.0;
        elasticEnergySquaredSum = 0.0;

        stepsInWindow =
            maxThermalizationSteps - thermalizationStep;

        if (
            stepsInWindow > POLYMER_WINDOW)
        {
            stepsInWindow = POLYMER_WINDOW;
        }

        for (
            windowStep = 0;
            windowStep < stepsInWindow;
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

            elasticEnergy =
                polymerElasticEnergy(
                    x,
                    y,
                    z,
                    polymer);
            elasticEnergySum += elasticEnergy;
            elasticEnergySquaredSum += elasticEnergy * elasticEnergy;
        }

        elasticEnergyMean =
            elasticEnergySum / stepsInWindow;

        elasticEnergyVariance =
            elasticEnergySquaredSum / stepsInWindow -
            elasticEnergyMean * elasticEnergyMean;

        if (
            elasticEnergyVariance < 0.0)
        {
            elasticEnergyVariance = 0.0;
        }

        elasticEnergyStd =
            sqrt(
                elasticEnergyVariance);

        printf(
            "Thermalization step %d: "
            "mean elastic energy = %.5f, "
            "std = %.5f\n",
            thermalizationStep,
            elasticEnergyMean,
            fabs(elasticEnergyStd-previousElasticEnergyStd)/
                (fabs(previousElasticEnergyStd) > 1e-10 ? fabs(previousElasticEnergyStd) : 1.0));

        if (
            thermalizationStep >= minThermalizationSteps &&
            hasPreviousWindow)
        {
            double meanScale =
                fmax(fabs(previousElasticEnergyMean), simulation.kBT);
            double stdScale =
                fmax(fabs(previousElasticEnergyStd), simulation.kBT);

            if (
                fabs(elasticEnergyMean - previousElasticEnergyMean) /
                    meanScale < tolerance &&
                fabs(elasticEnergyStd - previousElasticEnergyStd) /
                    stdScale < tolerance)
            {
                consecutiveWindows++;
            }

            else
            {
                consecutiveWindows = 0;
            }
        }

        else
        {
            consecutiveWindows = 0;
        }

        previousElasticEnergyMean = elasticEnergyMean;
        previousElasticEnergyStd = elasticEnergyStd;
        hasPreviousWindow = 1;

        if (
            consecutiveWindows >= requiredWindows)
        {
            thermalized = 1;
        }

    }

    if (
        thermalized)
    {
        printf(
            "Polymer thermalized after "
            "%d steps (t = %.3f)\n",
            thermalizationStep,
            thermalizationStep * simulation.dt);
    }

    else
    {
        printf(
            "Maximum thermalization time reached "
            "after %d steps (t = %.3f); continuing "
            "with production\n",
            thermalizationStep,
            thermalizationStep * simulation.dt);
    }

    file = fopen(
        outputFile,
        "w");

    if (
        file == NULL)
    {
        printf(
            "Error opening polymer "
            "output file\n");

        exit(1);
    }

    // Cabecera
    fprintf(
        file,
        "time,kinetic,elastic");

    for (
        int i = 0;
        i < polymer.N;
        i++)
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

            for (
                int i = 0;
                i < polymer.N;
                i++)
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

        if (
            step < numberOfSteps)
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

    fclose(
        file);

    parametersFile = fopen(
        "data/raw/polymer/parameters.txt",
        "w");

    if (
        parametersFile != NULL)
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
            "force=%lf\n",
            polymer.externalForce);

        fprintf(
            parametersFile,
            "firstEnd=%s\n",
            polymer.firstEndFixed ? "fixed" : "free");

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
            "minThermalizationTime=%lf\n",
            simulation.minThermalizationTime);

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
            "maxSavedPoints=%d\n",
            maxSavedPoints);

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