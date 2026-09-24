#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "integrators.hpp"

// Ejecuta la simulación completa
void runSimulation(
    int model,
    int algorithm,
    SimulationParameters simulation,
    ModelParameters modelParameters,
    char outputFile[]);

#endif