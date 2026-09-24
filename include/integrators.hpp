#pragma once

double rand_gaussian();

void guardar_parametros(
    double x,
    double p,
    double eta,
    double h,
    double t_max
);

void Euler_Maruyama(
    double x,
    double p,
    double eta,
    double h,
    double t_max
);