#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

int main(void) {



    return 0;
}


int main()
{
    std::cout << "TFIII - Simulacion Molecular\n";


    // Empezamos a poner los parametros necesarios para Euler-Maruyama en el OA
    
    double x, ŋ, p, h, t_max, Z;

    
    x = 1.0; // Posición inicial
    p = 0.0; // Momento inicial
    ŋ = 0.1; // Coeficiente de fricción
    h = 0.0001; // Paso de tiempo
    t_max = 10000; // Tiempo máximo de simulación


    rand_gaussian(); // Llamada inicial para establecer la semilla
    Euler_Maruyama(x, p, ŋ, h, t_max);

    std::cout << "Compilacion y ejecucion correctas.\n";



    return 0;
}