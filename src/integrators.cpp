#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "simulacion.h" // Importante: incluir tu propio .h con comillas

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif



void Euler_Maruyama(double x, double p, double ŋ, double h, double t_max)
{
    printf("Hola \n");
    double x_n, p_n, Z, E_c, E_p;
    // N_Int es el numero de medidas que queremos, si guardamos cada vez cuidado que acabas con 10Gbs en un archivo de texto
    int N_Int = 1000, contador = 0;
    double Int = (t_max / (h * N_Int));
    printf("Int: %.17f \n", Int);

    //Creamos archivo de texto para los datos con forma: t, x_n, p_n, E_c, E_p
    FILE*f;
    char filename[50];
    snprintf(filename, sizeof(filename), "datos_%.1f_%.4f.txt", ŋ, h);
    f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error al abrir datos.txt");
    return;
}

    
    for (double t = 0; t < t_max; t += h)
    {
        x_n = x + p * h;
        Z= sqrt(2 * ŋ * h)  * rand_gaussian();
        p_n = p + (-ŋ * p - x) * h + Z;

        x = x_n;
        p = p_n;


        E_c = 0.5 * p_n * p_n;
        E_p = 0.5 * x_n * x_n;

        contador++;
        if(contador == Int)
        {
            //printf("Resto para t=%f: %.17f \n", t, fmod(t, Int));
            fprintf(f, "%.17f %.17f %.17f %.17f %.17f \n", t, x_n, p_n, E_c, E_p);
            contador = 0;
        }


    }

    fclose(f);
    printf("Simulacion completada.\n");
}