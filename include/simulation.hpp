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