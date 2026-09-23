# TFIII - Simulación Molecular

Proyecto de la Actividad 5 de Técnicas Físicas III.

La simulación se desarrollará en C++ y el análisis de los datos se realizará posteriormente en Python. La estructura está pensada para reutilizar los mismos integradores con distintos modelos físicos.

## Estructura

```text
TFIII-Simulacion-Molecular/
│
├── .vscode/
│   ├── tasks.json
│   ├── launch.json
│   └── c_cpp_properties.json
│
├── include/
│   ├── simulation.hpp
│   ├── models.hpp
│   └── integrators.hpp
│
├── src/
│   ├── main.cpp
│   ├── simulation.cpp
│   ├── models.cpp
│   └── integrators.cpp
│
├── analysis/
│   └── harmonic_analysis.py
│
├── data/
│   └── raw/
│
├── results/
│   ├── figures/
│   └── tables/
│
├── bin/
├── .gitignore
└── README.md
```

## Archivos principales

### `src/main.cpp`

Punto de entrada del programa.

Aquí se definirán los parámetros de la simulación y se seleccionarán el modelo físico y el algoritmo de integración.

### `src/simulation.cpp`

Contiene el funcionamiento general de la simulación:

- evolución temporal;
- número de pasos;
- termalización;
- almacenamiento de datos;
- llamadas al integrador.

### `src/models.cpp`

Contiene las fuerzas y potenciales de los distintos sistemas físicos.

Inicialmente incluirá el oscilador armónico. Más adelante se añadirán el doble pozo y el polímero.

### `src/integrators.cpp`

Contiene los algoritmos de integración numérica:

- Euler-Maruyama;
- Runge-Kutta estocástico;
- Grønbech-Jensen.

## Carpeta `include/`

Contiene las cabeceras `.hpp` correspondientes a los archivos de `src/`.

- `simulation.hpp`: interfaz de la simulación.
- `models.hpp`: declaraciones de los modelos físicos.
- `integrators.hpp`: declaraciones de los algoritmos de integración.

## `analysis/harmonic_analysis.py`

Script de Python para analizar los datos obtenidos en la simulación del oscilador armónico y generar estadísticas y gráficas.

## `data/raw/`

Contiene los archivos generados directamente por las simulaciones en C++.

## `results/`

Contiene los resultados procesados:

- `figures/`: gráficas.
- `tables/`: tablas y resúmenes numéricos.

## `bin/`

Contiene el ejecutable generado al compilar el proyecto.

## `.vscode/`

Configuración de Visual Studio Code.

- `tasks.json`: indica cómo compilar el proyecto.
- `launch.json`: permite ejecutar y depurar desde VSCode.
- `c_cpp_properties.json`: configuración de C++ e IntelliSense.

Para trabajar correctamente, debe abrirse directamente la carpeta `TFIII-Simulacion-Molecular` como carpeta raíz en VSCode.