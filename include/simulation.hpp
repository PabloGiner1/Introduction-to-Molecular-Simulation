<<<<<<< HEAD
#pragma once

enum class Algorithm {
    EulerMaruyama,
    Runge_Kutta
};

struct Parameters {
    double x0;
    double p0;
    double eta;
    double h;
    double t_max;
    Algorithm algorithm;
};

void run_simulation(const Parameters& parameters);
=======
#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "integrators.hpp"

// Ejecuta la simulación completa
void runSimulation(int model, int algorithm, SimulationParameters simulation, ModelParameters modelParameters,
                   char outputFile[]);

#endif
>>>>>>> 6e59527563090f56cb92d37dd26ef7dfaeef89bd
