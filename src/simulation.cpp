#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ventana 5000

void runSimulation(
    int model,
    int algorithm,
    SimulationParameters simulation,
    ModelParameters modelParameters,
    char outputFile[])
{
    double x;
    double v;

    double time;
    double kineticEnergy;
    double potentialEnergy;

    double promedio_v;
    double promedio_v2;

    double desv_v;
    double desv_v_teorica;

    double varianza;
    double tolerancia;

    int numberOfSteps;
    int maxSavedPoints;
    int saveEvery;

    int minThermalizationSteps;

    int step;
    int step_termalizacion;

    int indice_ventana;
    int ventanas_consecutivas;

    int TermalizacionAlcanzada;

    FILE *file;
    FILE *parametersFile;

    x = simulation.initialPosition;
    v = simulation.initialVelocity;

    numberOfSteps =
        (int)(simulation.totalTime / simulation.dt);

    maxSavedPoints =
        simulation.maxSavedPoints;

    if (
        maxSavedPoints < 1)
    {
        maxSavedPoints = 10000;
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

    if (
        minThermalizationSteps > numberOfSteps)
    {
        printf(
            "Error: minimum thermalization "
            "time is larger than totalTime\n");

        exit(1);
    }

    file = fopen(
        outputFile,
        "w");

    if (
        file == NULL)
    {
        printf(
            "Error opening output file\n");

        exit(1);
    }

    fprintf(
        file,
        "time,x,v,kinetic,potential\n");

    // ===================================
    // Termalización
    // ===================================

    tolerancia = 0.05;

    desv_v_teorica = sqrt(
        simulation.kBT / simulation.mass);

    double v_ventana[ventana];

    int ventanas_necesarias = 3;

    indice_ventana = 0;
    ventanas_consecutivas = 0;

    step_termalizacion = 0;

    TermalizacionAlcanzada = 0;

    printf(
        "Thermalizing...\n");

    printf(
        "Minimum thermalization time: %g\n",
        simulation.minThermalizationTime);

    while (
        TermalizacionAlcanzada == 0)
    {
        if (
            step_termalizacion >= numberOfSteps)
        {
            printf(
                "Error: thermalization "
                "was not reached\n");

            fclose(
                file);

            exit(1);
        }

        integrationStep(
            algorithm,
            model,
            &x,
            &v,
            simulation,
            modelParameters);

        v_ventana[indice_ventana] = v;

        indice_ventana =
            (indice_ventana + 1) % ventana;

        step_termalizacion++;

        if (
            indice_ventana == 0)
        {
            promedio_v = 0.0;
            promedio_v2 = 0.0;

            for (
                int i = 0;
                i < ventana;
                i++)
            {
                promedio_v +=
                    v_ventana[i];

                promedio_v2 +=
                    v_ventana[i] * v_ventana[i];
            }

            promedio_v /=
                ventana;

            promedio_v2 /=
                ventana;

            varianza =
                promedio_v2 - promedio_v * promedio_v;

            if (
                varianza < 0.0)
            {
                varianza = 0.0;
            }

            desv_v =
                sqrt(
                    varianza);

            if (
                fabs(
                    (
                        desv_v - desv_v_teorica) /
                    desv_v_teorica) < tolerancia)
            {
                ventanas_consecutivas++;
            }

            else
            {
                ventanas_consecutivas = 0;
            }

            time =
                step_termalizacion * simulation.dt;

            if (
                ventanas_consecutivas >= ventanas_necesarias && step_termalizacion >= minThermalizationSteps)
            {
                printf(
                    "Thermalization reached "
                    "at step %d, t = %g\n",
                    step_termalizacion,
                    time);

                TermalizacionAlcanzada = 1;
            }
        }
    }

    // ===================================
    // Parámetros usados
    // ===================================

    parametersFile = fopen(
        "data/raw/harmonic/parameters.txt",
        "w");

    if (
        parametersFile != NULL)
    {
        fprintf(
            parametersFile,
            "x0=%lf\n",
            simulation.initialPosition);

        fprintf(
            parametersFile,
            "v0=%lf\n",
            simulation.initialVelocity);

        fprintf(
            parametersFile,
            "gamma=%lf\n",
            simulation.gamma);

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
            "mass=%lf\n",
            simulation.mass);

        fprintf(
            parametersFile,
            "kBT=%lf\n",
            simulation.kBT);

        fprintf(
            parametersFile,
            "k=%lf\n",
            modelParameters.k);

        fprintf(
            parametersFile,
            "algorithm=%d\n",
            algorithm);

        fprintf(
            parametersFile,
            "A=%lf\n",
            modelParameters.A);

        fprintf(
            parametersFile,
            "force=%lf\n",
            modelParameters.externalForce);

        fprintf(
            parametersFile,
            "model=%d\n",
            model);

        fprintf(
            parametersFile,
            "minThermalizationTime=%lf\n",
            simulation.minThermalizationTime);

        fprintf(
            parametersFile,
            "thermalizationSteps=%d\n",
            step_termalizacion);

        fprintf(
            parametersFile,
            "thermalizationTime=%lf\n",
            step_termalizacion * simulation.dt);

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

    // ===================================
    // Producción
    // ===================================

    printf(
        "Production run...\n");

    for (
        step = 0;
        step <= numberOfSteps;
        step++)
    {
        time =
            step * simulation.dt;

        if (
            step < numberOfSteps)
        {
            integrationStep(
                algorithm,
                model,
                &x,
                &v,
                simulation,
                modelParameters);
        }

        if (
            step % saveEvery == 0 || step == numberOfSteps)
        {
            kineticEnergy =
                0.5 * simulation.mass * v * v;

            potentialEnergy =
                potential(
                    model,
                    x,
                    modelParameters);

            fprintf(
                file,
                "%lf,%lf,%lf,%lf,%lf\n",
                time,
                x,
                v,
                kineticEnergy,
                potentialEnergy);
        }
    }

    fclose(
        file);

    printf(
        "Saved time resolution = %g\n",
        saveEvery * simulation.dt);
}