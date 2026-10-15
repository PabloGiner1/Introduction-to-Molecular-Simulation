#include "simulation.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ventana 2000

void runSimulation(int model, int algorithm, SimulationParameters simulation, ModelParameters modelParameters,
                   char outputFile[])
{
    double x;
    double v;

    double time;
    double kineticEnergy;
    double potentialEnergy;

    double promedio_v = 0.0;
    double promedio_v2 = 0.0;
    double desv_v;
    double desv_v_teorica;
    double varianza;
    double tolerancia;

    int numberOfSteps;
    int maxSavedPoints;
    int saveEvery;
    int step;
    int TermalizacionAlcanzada = 0;

    FILE *file;
    FILE *parametersFile;

    // Condiciones iniciales
    x = simulation.initialPosition;
    v = simulation.initialVelocity;

    // Número total de pasos
    numberOfSteps = (int)(simulation.totalTime / simulation.dt);

    // Guardamos como máximo unas 10000 medidas
    maxSavedPoints = 10000;
    saveEvery = numberOfSteps / maxSavedPoints;

    if (saveEvery < 1)
    {
        saveEvery = 1;
    }

    // Abrimos el archivo de resultados
    file = fopen(outputFile, "w");

    if (file == NULL)
    {
        printf("Error opening output file\n");
        exit(1);
    }

    // Guardamos los parámetros usados
    parametersFile = fopen(
        "data/raw/harmonic/parameters.txt",
        "w");

    if (parametersFile != NULL)
    {
        fprintf(parametersFile, "x0=%lf\n", simulation.initialPosition);
        fprintf(parametersFile, "v0=%lf\n", simulation.initialVelocity);
        fprintf(parametersFile, "gamma=%lf\n", simulation.gamma);
        fprintf(parametersFile, "dt=%lf\n", simulation.dt);
        fprintf(parametersFile, "totalTime=%lf\n", simulation.totalTime);
        fprintf(parametersFile, "mass=%lf\n", simulation.mass);
        fprintf(parametersFile, "kBT=%lf\n", simulation.kBT);
        fprintf(parametersFile, "k=%lf\n", modelParameters.k);
        fprintf(parametersFile, "algorithm=%d\n", algorithm);
        fprintf(parametersFile, "A=%lf\n", modelParameters.A);
        fprintf(parametersFile, "model=%d\n", model);


        fclose(parametersFile);
    }

    // Cabecera del CSV
    fprintf(file, "time,x,v,kinetic,potential\n");


    // Valores teoricos para termalizacion
    tolerancia = 0.05;
    desv_v_teorica = sqrt(simulation.kBT / simulation.mass);
    double v_ventana[ventana];
    int indice_ventana = 0;
    int ventanas_consecutivas = 0;
    int ventanas_permitidas = 0;
    int ventanas_necesarias =3;



    // Bucle temporal
    for (step = 0; step <= numberOfSteps; step++)
    {
        time = step * simulation.dt;

        if (step < numberOfSteps)
        {
            integrationStep(
                algorithm,
                model,
                &x,
                &v,
                simulation,
                modelParameters);
        }

        //Hacemos una ventana movil, cuando llega a 500 sustituye el valor inicial por el siguiente y así sucesivamente
        v_ventana[indice_ventana] = v;
        indice_ventana = (indice_ventana + 1) % ventana;
        if(indice_ventana == 0)
        {
            //Calculamos la media y la varianza de la velocidad para determinar si se ha alcanzado la termalización
            promedio_v = 0.0;
            promedio_v2 = 0.0;
            for(int i = 0; i < ventana; i++)
            {
                promedio_v += v_ventana[i];
                promedio_v2 += v_ventana[i] * v_ventana[i];
            }
            varianza = promedio_v2 / ventana - (promedio_v / ventana) * (promedio_v / ventana);
            desv_v = sqrt(varianza);
        }




        if(step > ventana && TermalizacionAlcanzada == 0 && indice_ventana == 0)
        {
            if(fabs((desv_v - desv_v_teorica)/desv_v_teorica) < tolerancia){
            ventanas_consecutivas++;
            }else {
                ventanas_permitidas++;
                if(ventanas_permitidas > 1)
                {
                ventanas_consecutivas = 0;
                ventanas_permitidas = 0;
                }
            }
            if(ventanas_consecutivas >= ventanas_necesarias)
            {
                printf("Termalizacion alcanzada en el tiempo %lf\n", time);
                TermalizacionAlcanzada = 1;
            }
            
        }

                if ((step % saveEvery == 0 || step == numberOfSteps) && TermalizacionAlcanzada == 1)
        {
            kineticEnergy = 0.5 * simulation.mass * v * v;
            potentialEnergy = potential(model, x, modelParameters);

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

    fclose(file);
}