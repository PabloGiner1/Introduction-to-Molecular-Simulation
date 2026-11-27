#include "polymer.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define POLYMER_WINDOW 200000
#define POLYMER_DECORRELATION_GAP 200000
#define POLYMER_MAX_THERMALIZATION_TIME 1000000.0

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
    double meanKineticEnergy;
    double meanElasticEnergy;
    double kineticEnergySum;
    double elasticEnergySum;
    double elasticEnergyMean;
    double elasticEnergyStd;
    double previousElasticEnergyMean;
    double kineticEnergyMean;
    double kineticEnergyM2;
    double kineticEnergyStd;
    double previousKineticEnergyMean;
    double elasticEnergyM2;

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

    int productionStepCount;

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
        stepsInWindow =
            maxThermalizationSteps - thermalizationStep;

        if (
            stepsInWindow > POLYMER_WINDOW)
        {
            stepsInWindow = POLYMER_WINDOW;
        }

        elasticEnergyMean = 0.0;
        elasticEnergyM2 = 0.0;
        kineticEnergyMean = 0.0;
        kineticEnergyM2 = 0.0;

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
            kineticEnergy =
                polymerKineticEnergy(
                    vx,
                    vy,
                    vz,
                    simulation,
                    polymer);

            int sampleCount =
                windowStep + 1;

            double elasticDelta =
                elasticEnergy - elasticEnergyMean;
            elasticEnergyMean +=
                elasticDelta / sampleCount;
            elasticEnergyM2 +=
                elasticDelta *
                (elasticEnergy - elasticEnergyMean);

            double kineticDelta =
                kineticEnergy - kineticEnergyMean;
            kineticEnergyMean +=
                kineticDelta / sampleCount;
            kineticEnergyM2 +=
                kineticDelta *
                (kineticEnergy - kineticEnergyMean);
        }

        elasticEnergyStd =
            stepsInWindow > 1
                ? sqrt(elasticEnergyM2 / (stepsInWindow - 1))
                : 0.0;
        kineticEnergyStd =
            stepsInWindow > 1
                ? sqrt(kineticEnergyM2 / (stepsInWindow - 1))
                : 0.0;

        printf(
            "Thermalization step %d: "
            "elastic mean/std = %.5f/%.5f, "
            "kinetic mean/std = %.5f/%.5f\n",
            thermalizationStep,
            elasticEnergyMean,
            elasticEnergyStd,
            kineticEnergyMean,
            kineticEnergyStd);

        if (
            thermalizationStep >= minThermalizationSteps &&
            hasPreviousWindow)
        {
            double elasticMeanScale =
                fmax(fabs(previousElasticEnergyMean), simulation.kBT);
            double kineticMeanScale =
                fmax(fabs(previousKineticEnergyMean), simulation.kBT);
            double elasticMeanChange =
                fabs(elasticEnergyMean - previousElasticEnergyMean) /
                elasticMeanScale;
            double kineticMeanChange =
                fabs(kineticEnergyMean - previousKineticEnergyMean) /
                kineticMeanScale;

            int elasticStable =
                elasticMeanChange < tolerance;
            int kineticStable =
                kineticMeanChange < tolerance;

            printf(
                "  Relative mean changes: elastic=%.3f%%, "
                "kinetic=%.3f%%; "
                "stable windows=%d/%d\n",
                100.0 * elasticMeanChange,
                100.0 * kineticMeanChange,
                (elasticStable && kineticStable)
                    ? consecutiveWindows + 1
                    : 0,
                requiredWindows);

            if (elasticStable && kineticStable)
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
        previousKineticEnergyMean = kineticEnergyMean;
        hasPreviousWindow = 1;

        if (
            consecutiveWindows >= requiredWindows)
        {
            thermalized = 1;
        }
        else
        {
            int gapRemaining =
                POLYMER_DECORRELATION_GAP;

            if (
                maxThermalizationSteps - thermalizationStep <
                gapRemaining)
            {
                gapRemaining =
                    maxThermalizationSteps -
                    thermalizationStep;
            }

            for (
                int gapStep = 0;
                gapStep < gapRemaining &&
                thermalizationStep < maxThermalizationSteps;
                gapStep++)
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
            }
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
        "time,kinetic,elastic,meanKinetic,meanElastic");

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
    kineticEnergySum = 0.0;
    elasticEnergySum = 0.0;
    productionStepCount = 0;

    for (
        int step = 0;
        step <= numberOfSteps;
        step++)
    {
        time =
            step * simulation.dt;

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

        kineticEnergySum += kineticEnergy;
        elasticEnergySum += elasticEnergy;
        productionStepCount++;

        meanKineticEnergy =
            kineticEnergySum / productionStepCount;
        meanElasticEnergy =
            elasticEnergySum / productionStepCount;

        if (
            step % saveEvery == 0 || step == numberOfSteps)
        {
            fprintf(
                file,
                "%.10f,%.10f,%.10f,%.10f,%.10f",
                time,
                kineticEnergy,
                elasticEnergy,
                meanKineticEnergy,
                meanElasticEnergy);

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