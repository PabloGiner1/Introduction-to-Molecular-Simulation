#include "simulation.hpp"
#include "integrators.hpp"

void run_simulation(const Parameters& parameters) {
    switch (parameters.algorithm) {
        case Algorithm::EulerMaruyama:
            Euler_Maruyama(
                parameters.x0,
                parameters.p0,
                parameters.eta,
                parameters.h,
                parameters.t_max
            );
            break;

        case Algorithm::Runge_Kutta:
            // Runge-Kutta y Verlet estocastico...

            break;
    }
}