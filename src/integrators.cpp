#include "integrators.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
<<<<<<< HEAD
#include <time.h>
#include <filesystem>
#include <fstream>


#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif



void guardar_parametros(double x, double p, double eta, double h, double t_max)
{
    std::filesystem::create_directories("data/raw");

    std::ofstream file("data/raw/parameters.txt");
    file << "x=" << x << '\n';
    file << "p=" << p << '\n';
    file << "eta=" << eta << '\n';
    file << "h=" << h << '\n';
    file << "t_max=" << t_max << '\n';
}

void Euler_Maruyama(double x, double p, double eta, double h, double t_max)
{
    printf("Hola \n");
    double x_n, p_n, Z, E_c, E_p;


    // N_Int es el numero de medidas que queremos, si guardamos cada vez cuidado que acabas con 10Gbs en un archivo de texto depende los parametros
    int N_Int = 10000, contador = 0;
    double Int = (t_max / (h * N_Int));
    printf("Intervalo: %.0f \n", Int);

    //Creamos archivo de texto para los datos con forma: t, x_n, p_n, E_c, E_p
    FILE*f;
    char filename[50];
    std::filesystem::create_directories("data/raw");
    snprintf(filename, sizeof(filename), "data/raw/datos_%.1f_%.4f.txt", eta, h);
    f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error al abrir datos.txt");
    return;
}

    
    for (double t = 0; t < t_max; t += h)
    {
        x_n = x + p * h;
        Z= sqrt(2 * eta * h)  * rand_gaussian();
        p_n = p + (-eta * p - x) * h + Z;

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

// Genera un número con distribución gaussiana N(0, 1)
double rand_gaussian(void) {
    static int tiene_guardado = 0;
    static double z1;

    // La transformación genera 2 números a la vez, guardamos uno para la siguiente llamada
    if (tiene_guardado) {
        tiene_guardado = 0;
        return z1;
    }

    double u1, u2;
    // u1 debe ser estrictamente mayor que 0 para evitar log(0)
    do {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 <= 0.0);
    
    u2 = (double)rand() / RAND_MAX;

    // Transformación de Box-Muller
    double radio = sqrt(-2.0 * log(u1));
    double z0 = radio * cos(2.0 * M_PI * u2);
    z1 = radio * sin(2.0 * M_PI * u2);

    tiene_guardado = 1;
=======

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void integrationStep(int algorithm, int model, double *x, double *v, SimulationParameters simulation,
                     ModelParameters modelParameters)
{

    if (algorithm == EULER_MARUYAMA)
    {
        eulerMaruyamaStep(model, x, v, simulation, modelParameters);
        return;
    }

    if (algorithm == RUNGE_KUTTA)
    {
        printf("Runge-Kutta not implemented yet\n");
        exit(1);
    }

    if (algorithm == GJF)
    {
        printf("G-JF not implemented yet\n");
        exit(1);
    }

    // Número de algoritmo incorrecto
    printf("Error: unknown integration algorithm\n");
    exit(1);
}

// Realiza un paso de Euler-Maruyama
void eulerMaruyamaStep(int model, double *x, double *v, SimulationParameters simulation,
                       ModelParameters modelParameters)
{
    double xOld;
    double vOld;
    double xNew;
    double vNew;
    double F;
    double Z;

    // Guardamos la posición y velocidad actuales
    xOld = *x;
    vOld = *v;

    // Calculamos la fuerza en la posición actual
    F = force(model, xOld, modelParameters);

    // Generamos el ruido gaussiano
    Z = randGaussian();

    // Actualizamos la posición
    xNew = xOld + vOld * simulation.dt;

    // Actualizamos la velocidad
    vNew = vOld + (F - simulation.gamma * vOld) * simulation.dt / simulation.mass + sqrt(2.0 * simulation.gamma * simulation.kBT * simulation.dt) * Z / simulation.mass;

    // Guardamos los nuevos valores
    *x = xNew;
    *v = vNew;
}

// Genera un número aleatorio gaussiano N(0,1) con Box-Muller
double randGaussian()
{
    static int saved = 0;
    static double z1;

    double u1;
    double u2;
    double radius;
    double z0;

    // Box-Muller genera dos gaussianos; reutilizamos el segundo
    if (saved == 1)
    {
        saved = 0;
        return z1;
    }

    // u1 debe ser mayor que 0 para evitar log(0)
    do
    {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 <= 0.0);

    u2 = (double)rand() / RAND_MAX;

    // Transformación de Box-Muller
    radius = sqrt(-2.0 * log(u1));
    z0 = radius * cos(2.0 * M_PI * u2);
    z1 = radius * sin(2.0 * M_PI * u2);

    saved = 1;

>>>>>>> 6e59527563090f56cb92d37dd26ef7dfaeef89bd
    return z0;
}