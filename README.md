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

Script de Python para analizar los datos de una ejecución y generar estadísticas y gráficas. Para el doble pozo también calcula y representa la ocupación acumulada de los estados `x < 0` y `x > 0` en función del tiempo, y la distribución de los tiempos consecutivos de estancia en cada pozo. La distribución usa episodios observados entre dos cambios de estado; excluye los intervalos que ya estaban en curso al inicio o que siguen abiertos al final de la simulación. Las muestras con `x = 0` no se asignan a ningún pozo y separan episodios.

## `analysis/parameters_sweep.py`

Ejecuta el análisis detallado del doble pozo para cada combinación de `A = 0.5, 1.0, 2.0`, `eta = 0.01, 0.1, 1.0, 2.0, 5.0` y posición inicial `x0 = -1, 0, 1`, usando el algoritmo, el paso temporal y el tiempo total definidos en `config/simulation.txt`. Produce 45 simulaciones y guarda sus gráficas en `results/double_well/parameter_sweep/single_run/figures/`, separadas por `A` y `eta`, sin mezclarlas con `single_run` normal. Esa carpeta se limpia al iniciar cada barrido, por lo que las salidas anteriores del barrido se reemplazan. Los nombres de las gráficas incluyen algoritmo, `x0` y fecha, sin `eta` ni `h`. El archivo de configuración se restaura al terminar. En VS Code se puede iniciar desde la configuración de lanzamiento o la tarea `Barrido detallado pozo doble: A, eta y x0`; ambas compilan primero la simulación.

## `data/raw/`

Contiene los archivos generados directamente por las simulaciones en C++.

## `results/`

Contiene los resultados procesados:

- `figures/`: gráficas, organizadas en subcarpetas por `A` y después por `eta`; cada archivo incluye fecha y hora para conservar todas las ejecuciones.
- `tables/`: tablas y resúmenes numéricos.

## `bin/`

Contiene el ejecutable generado al compilar el proyecto.

## `.vscode/`

Configuración de Visual Studio Code.

- `tasks.json`: indica cómo compilar el proyecto.
- `launch.json`: permite ejecutar y depurar desde VSCode.
- `c_cpp_properties.json`: configuración de C++ e IntelliSense.

Para trabajar correctamente, debe abrirse directamente la carpeta `TFIII-Simulacion-Molecular` como carpeta raíz en VSCode.