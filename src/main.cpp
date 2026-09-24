#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "integrators.hpp"
#include"simulation.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

int main(void) {

    // Empezamos a poner los parametros necesarios para Euler-Maruyama en el OA
    
    std::cout << "TFIII - Simulacion Molecular\n";

    Parameters parametros;
 
    parametros.x0 = 1.0;                                            // Posición inicial
    parametros.p0 = 0.0;                                            // Momento inicial
    parametros.eta = 0.1;                                           // Amortiguamiento: 0.1, 1.0, 10
    parametros.h = 0.1;                                          // Paso de tiempo: 0.1, 0.001, 0.0001, 0.00001
    parametros.t_max = 10000.0;                                     // Tiempo máximo de simulación
    parametros.algorithm = Algorithm::EulerMaruyama;                //Alrgoritmo de integración: Euler-Maruyama, Runge-Kutta, Verlet



    rand_gaussian(); // Llamada inicial para establecer la semilla
    guardar_parametros(
        parametros.x0,
        parametros.p0,
        parametros.eta,
        parametros.h,
        parametros.t_max
    );
    printf("Parametros guardados en data/raw/parameters.txt\n");

    run_simulation(parametros);
    std::cout << "Compilacion y ejecucion correctas.\n";

    return 0;
}