from pathlib import Path

import csv
import os
import secrets
import subprocess

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]

config_file = (
    project_root
    / "config"
    / "simulation.txt"
)

parameters_file = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
    / "parameters.txt"
)

data_folder = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
)

results_folder = (
    project_root
    / "results"
    / "double_well"
    / "force_sweep"
)

tables_folder = (
    results_folder
    / "tables"
)

figures_folder = (
    results_folder
    / "figures"
)


tables_folder.mkdir(
    parents=True,
    exist_ok=True
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


# Valores de fuerza externa
force_values = np.array([
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4
])


# Condiciones iniciales independientes
initial_positions = [
    -1.0,
    0.0,
    1.0
]


# Fuerzas que se mostrarán
# en la gráfica del potencial
potential_plot_forces = [
    -0.2,
    0.0,
    0.2
]


algorithms = {
    "EULER_MARUYAMA": "euler",
    "RUNGE_KUTTA": "rk",
    "GJF": "gjf"
}


if os.name == "nt":

    simulation_program = (
        project_root
        / "bin"
        / "simulation.exe"
    )

else:

    simulation_program = (
        project_root
        / "bin"
        / "simulation"
    )


if not simulation_program.exists():

    raise RuntimeError(
        "No se encuentra el ejecutable. "
        "Compila primero la simulacion."
    )


def parse_key_values(text):

    values = {}

    for line in text.splitlines():

        content = (
            line
            .split(
                "#",
                maxsplit=1
            )[0]
            .strip()
        )

        if "=" in content:

            name, value = (
                content.split(
                    "=",
                    maxsplit=1
                )
            )

            values[
                name.strip()
            ] = value.strip()

    return values


def integrate(y, x):

    # Compatible con distintas
    # versiones de NumPy
    if hasattr(
        np,
        "trapezoid"
    ):

        return np.trapezoid(
            y,
            x
        )

    return np.trapz(
        y,
        x
    )


def theoretical_occupancy(
    A,
    force,
    kBT
):

    x = np.linspace(
        -3.0,
        3.0,
        60001
    )


    potential = (
        A
        * (
            x ** 2
            - 1.0
        ) ** 2
        - force * x
    )


    weights = np.exp(
        -potential
        / kBT
    )


    total = integrate(
        weights,
        x
    )


    left_mask = (
        x < 0.0
    )

    right_mask = (
        x > 0.0
    )


    probability_left = (
        integrate(
            weights[left_mask],
            x[left_mask]
        )
        / total
    )


    probability_right = (
        integrate(
            weights[right_mask],
            x[right_mask]
        )
        / total
    )


    return (
        probability_left,
        probability_right
    )


def residence_statistics(
    time,
    positions
):

    # Estado -1: pozo izquierdo
    # Estado +1: pozo derecho
    states = np.where(
        positions >= 0.0,
        1,
        -1
    )


    occupation_right = float(
        np.mean(
            states == 1
        )
    )

    occupation_left = float(
        np.mean(
            states == -1
        )
    )


    transitions = int(
        np.count_nonzero(
            states[1:]
            != states[:-1]
        )
    )


    left_residence_times = []
    right_residence_times = []


    current_state = (
        states[0]
    )

    residence_start = (
        time[0]
    )


    # El primer intervalo puede haber
    # empezado antes del inicio del CSV,
    # así que no lo utilizamos.
    first_interval = True


    for i in range(
        1,
        len(states)
    ):

        if (
            states[i]
            != current_state
        ):

            duration = (
                time[i]
                - residence_start
            )


            if not first_interval:

                if current_state == 1:

                    right_residence_times.append(
                        duration
                    )

                else:

                    left_residence_times.append(
                        duration
                    )


            else:

                first_interval = False


            current_state = (
                states[i]
            )

            residence_start = (
                time[i]
            )


    # El último intervalo tampoco se utiliza
    # porque queda truncado al acabar la simulación.


    left_residence_times = np.array(
        left_residence_times,
        dtype=float
    )

    right_residence_times = np.array(
        right_residence_times,
        dtype=float
    )


    if len(
        left_residence_times
    ) > 0:

        mean_left_residence = float(
            np.mean(
                left_residence_times
            )
        )

        median_left_residence = float(
            np.median(
                left_residence_times
            )
        )

    else:

        mean_left_residence = np.nan
        median_left_residence = np.nan


    if len(
        right_residence_times
    ) > 0:

        mean_right_residence = float(
            np.mean(
                right_residence_times
            )
        )

        median_right_residence = float(
            np.median(
                right_residence_times
            )
        )

    else:

        mean_right_residence = np.nan
        median_right_residence = np.nan


    if len(time) > 1:

        saved_dt = float(
            np.median(
                np.diff(
                    time
                )
            )
        )

    else:

        saved_dt = np.nan


    return {
        "occupation_left":
            occupation_left,

        "occupation_right":
            occupation_right,

        "transitions":
            transitions,

        "mean_residence_left":
            mean_left_residence,

        "mean_residence_right":
            mean_right_residence,

        "median_residence_left":
            median_left_residence,

        "median_residence_right":
            median_right_residence,

        "residence_count_left":
            len(
                left_residence_times
            ),

        "residence_count_right":
            len(
                right_residence_times
            ),

        "saved_dt":
            saved_dt
    }


def mean_and_std(
    values
):

    values = np.array(
        values,
        dtype=float
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:

        return (
            np.nan,
            np.nan
        )


    mean = float(
        np.mean(
            values
        )
    )


    if len(values) > 1:

        std = float(
            np.std(
                values,
                ddof=1
            )
        )

    else:

        std = 0.0


    return (
        mean,
        std
    )


original_config = (
    config_file.read_text()
)


base_config = (
    parse_key_values(
        original_config
    )
)


algorithm = (
    base_config.get(
        "algorithm",
        "GJF"
    )
)


if algorithm not in algorithms:

    raise ValueError(
        f"Algoritmo desconocido: "
        f"{algorithm}"
    )


eta = float(
    base_config.get(
        "eta",
        "1.0"
    )
)

A = float(
    base_config.get(
        "A",
        "1.0"
    )
)

dt = float(
    base_config.get(
        "h",
        "0.01"
    )
)

total_time = float(
    base_config.get(
        "totalTime",
        "100000"
    )
)


algorithm_name = (
    algorithms[
        algorithm
    ]
)


data_file = (
    data_folder
    / f"harmonic_{algorithm_name}.csv"
)


base_seed = secrets.randbelow(
    (2 ** 32 - 1)
    - 30000
)


results = []


try:

    for force_value in force_values:

        print()
        print(
            "==================================="
        )

        print(
            f"F = {force_value:g}"
        )

        print(
            "==================================="
        )


        for initial_index, x0 in enumerate(
            initial_positions
        ):

            seed = (
                base_seed
                + initial_index
                * 11111
            )


            config_file.write_text(
                f"algorithm={algorithm}\n"
                f"eta={eta:.12g}\n"
                f"A={A:.12g}\n"
                f"force={force_value:.12g}\n"
                f"h={dt:.12g}\n"
                f"totalTime={total_time:.12g}\n"
                f"seed={seed}\n"
                f"x0={x0:.12g}\n"
                f"v0=0.0\n"
            )


            print(
                f"Simulando x0={x0:g}, "
                f"seed={seed}"
            )


            simulation_result = (
                subprocess.run(
                    [
                        str(
                            simulation_program
                        )
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
            )


            if (
                simulation_result.returncode
                != 0
            ):

                raise RuntimeError(
                    "Fallo en la simulacion "
                    f"para F={force_value:g}, "
                    f"x0={x0:g}\n"
                    f"{simulation_result.stdout}\n"
                    f"{simulation_result.stderr}"
                )


            parameters = (
                parse_key_values(
                    parameters_file
                    .read_text()
                )
            )


            kBT = float(
                parameters[
                    "kBT"
                ]
            )


            actual_A = float(
                parameters.get(
                    "A",
                    A
                )
            )


            actual_eta = float(
                parameters.get(
                    "gamma",
                    eta
                )
            )


            data = np.loadtxt(
                data_file,
                delimiter=",",
                skiprows=1
            )


            data = np.atleast_2d(
                data
            )


            if (
                len(data) < 2
                or not np.all(
                    np.isfinite(
                        data
                    )
                )
            ):

                raise RuntimeError(
                    "Datos no validos para "
                    f"F={force_value:g}, "
                    f"x0={x0:g}"
                )


            time = (
                data[
                    :,
                    0
                ]
            )

            positions = (
                data[
                    :,
                    1
                ]
            )


            statistics = (
                residence_statistics(
                    time,
                    positions
                )
            )


            theory_left, theory_right = (
                theoretical_occupancy(
                    actual_A,
                    force_value,
                    kBT
                )
            )


            results.append({
                "force":
                    force_value,

                "x0":
                    x0,

                "seed":
                    seed,

                "A":
                    actual_A,

                "eta":
                    actual_eta,

                "kBT":
                    kBT,

                "occupation_left":
                    statistics[
                        "occupation_left"
                    ],

                "occupation_right":
                    statistics[
                        "occupation_right"
                    ],

                "theory_left":
                    theory_left,

                "theory_right":
                    theory_right,

                "transitions":
                    statistics[
                        "transitions"
                    ],

                "mean_residence_left":
                    statistics[
                        "mean_residence_left"
                    ],

                "mean_residence_right":
                    statistics[
                        "mean_residence_right"
                    ],

                "median_residence_left":
                    statistics[
                        "median_residence_left"
                    ],

                "median_residence_right":
                    statistics[
                        "median_residence_right"
                    ],

                "residence_count_left":
                    statistics[
                        "residence_count_left"
                    ],

                "residence_count_right":
                    statistics[
                        "residence_count_right"
                    ],

                "saved_dt":
                    statistics[
                        "saved_dt"
                    ]
            })


            print(
                "  P(-) = "
                f"{statistics['occupation_left']:.3f}, "
                "P(+) = "
                f"{statistics['occupation_right']:.3f}"
            )

            print(
                "  Transiciones observadas = "
                f"{statistics['transitions']}"
            )


finally:

    config_file.write_text(
        original_config
    )


if len(results) == 0:

    raise RuntimeError(
        "No se han generado resultados."
    )


A_used = (
    results[0]["A"]
)

eta_used = (
    results[0]["eta"]
)

kBT_used = (
    results[0]["kBT"]
)


summary = []


for force_value in force_values:

    rows = [
        row
        for row in results
        if np.isclose(
            row["force"],
            force_value
        )
    ]


    left_values = [
        row[
            "occupation_left"
        ]
        for row in rows
    ]


    right_values = [
        row[
            "occupation_right"
        ]
        for row in rows
    ]


    residence_left_values = [
        row[
            "mean_residence_left"
        ]
        for row in rows
    ]


    residence_right_values = [
        row[
            "mean_residence_right"
        ]
        for row in rows
    ]


    transition_values = [
        row[
            "transitions"
        ]
        for row in rows
    ]


    occupation_left_mean, occupation_left_std = (
        mean_and_std(
            left_values
        )
    )


    occupation_right_mean, occupation_right_std = (
        mean_and_std(
            right_values
        )
    )


    residence_left_mean, residence_left_std = (
        mean_and_std(
            residence_left_values
        )
    )


    residence_right_mean, residence_right_std = (
        mean_and_std(
            residence_right_values
        )
    )


    transitions_mean, transitions_std = (
        mean_and_std(
            transition_values
        )
    )


    summary.append({
        "force":
            force_value,

        "occupation_left_mean":
            occupation_left_mean,

        "occupation_left_std":
            occupation_left_std,

        "occupation_right_mean":
            occupation_right_mean,

        "occupation_right_std":
            occupation_right_std,

        "theory_left":
            rows[0][
                "theory_left"
            ],

        "theory_right":
            rows[0][
                "theory_right"
            ],

        "mean_residence_left":
            residence_left_mean,

        "std_residence_left":
            residence_left_std,

        "mean_residence_right":
            residence_right_mean,

        "std_residence_right":
            residence_right_std,

        "transitions_mean":
            transitions_mean,

        "transitions_std":
            transitions_std,

        "saved_dt_mean":
            float(
                np.mean([
                    row[
                        "saved_dt"
                    ]
                    for row in rows
                ])
            )
    })


individual_file = (
    tables_folder
    / "double_well_force_sweep.csv"
)


with individual_file.open(
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            results[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        results
    )


summary_file = (
    tables_folder
    / "double_well_force_summary.csv"
)


with summary_file.open(
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            summary[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        summary
    )


forces = np.array([
    row[
        "force"
    ]
    for row in summary
])


right_mean = np.array([
    row[
        "occupation_right_mean"
    ]
    for row in summary
])


right_std = np.array([
    row[
        "occupation_right_std"
    ]
    for row in summary
])


left_mean = np.array([
    row[
        "occupation_left_mean"
    ]
    for row in summary
])


left_std = np.array([
    row[
        "occupation_left_std"
    ]
    for row in summary
])


theory_right = np.array([
    row[
        "theory_right"
    ]
    for row in summary
])


theory_left = np.array([
    row[
        "theory_left"
    ]
    for row in summary
])


# ===================================
# 1. Ocupación frente a fuerza
# ===================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    forces,
    theory_right,
    linestyle="--",
    label=r"Teoría $P(x>0)$"
)


plt.plot(
    forces,
    theory_left,
    linestyle="--",
    label=r"Teoría $P(x<0)$"
)


plt.axhline(
    0.5,
    linestyle=":",
    label="Ocupación 1/2"
)


plt.axvline(
    0.0,
    linestyle=":"
)


plt.errorbar(
    forces,
    right_mean,
    yerr=right_std,
    marker="o",
    capsize=4,
    label=r"Simulación $P(x>0)$"
)


plt.errorbar(
    forces,
    left_mean,
    yerr=left_std,
    marker="o",
    capsize=4,
    label=r"Simulación $P(x<0)$"
)


plt.xlabel(
    "Fuerza externa $F$"
)

plt.ylabel(
    "Ocupación"
)

plt.ylim(
    -0.02,
    1.02
)


plt.title(
    "Ocupación de los pozos frente a la fuerza externa\n"
    f"$A={A_used:g}$, "
    f"$\\eta={eta_used:g}$, "
    f"$k_BT={kBT_used:g}$"
)


plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figures_folder
    / "double_well_occupation_vs_force.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 2. Potencial para varias fuerzas
# ===================================

x_plot = np.linspace(
    -1.8,
    1.8,
    1200
)


plt.figure(
    figsize=(8, 5)
)


for force_value in potential_plot_forces:

    potential = (
        A_used
        * (
            x_plot ** 2
            - 1.0
        ) ** 2
        - force_value
        * x_plot
    )


    plt.plot(
        x_plot,
        potential,
        label=fr"$F={force_value:g}$"
    )


plt.axvline(
    0.0,
    linestyle=":",
    alpha=0.7
)


plt.xlabel(
    "Posición $x$"
)

plt.ylabel(
    "Potencial $V(x)$"
)


plt.title(
    "Inclinación del doble pozo por una fuerza externa\n"
    f"$A={A_used:g}$"
)


plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figures_folder
    / "double_well_potential_vs_force.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 3. Tiempo medio de estancia
# ===================================

residence_left = np.array([
    row[
        "mean_residence_left"
    ]
    for row in summary
])


residence_left_std = np.array([
    row[
        "std_residence_left"
    ]
    for row in summary
])


residence_right = np.array([
    row[
        "mean_residence_right"
    ]
    for row in summary
])


residence_right_std = np.array([
    row[
        "std_residence_right"
    ]
    for row in summary
])


plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    forces,
    residence_right,
    yerr=residence_right_std,
    marker="o",
    capsize=4,
    label=r"$\langle\tau_+\rangle$"
)


plt.errorbar(
    forces,
    residence_left,
    yerr=residence_left_std,
    marker="o",
    capsize=4,
    label=r"$\langle\tau_-\rangle$"
)


plt.axvline(
    0.0,
    linestyle=":"
)


plt.xlabel(
    "Fuerza externa $F$"
)

plt.ylabel(
    "Tiempo medio de estancia"
)


plt.title(
    "Tiempo medio de estancia en cada pozo\n"
    f"$A={A_used:g}$, "
    f"$\\eta={eta_used:g}$"
)


# Los tiempos pueden cambiar
# varios órdenes de magnitud.
plt.yscale(
    "log"
)


plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figures_folder
    / "double_well_residence_time_vs_force.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 4. Logaritmo del cociente de ocupaciones
# ===================================

valid_simulation = (
    (right_mean > 0.0)
    & (left_mean > 0.0)
)


simulation_log_ratio = np.full(
    len(forces),
    np.nan
)


simulation_log_ratio[
    valid_simulation
] = np.log(
    right_mean[
        valid_simulation
    ]
    / left_mean[
        valid_simulation
    ]
)


# Propagación aproximada de la
# dispersión de P+ al logit.
log_ratio_std = np.full(
    len(forces),
    np.nan
)


for i in range(
    len(forces)
):

    p = right_mean[i]

    sigma_p = right_std[i]


    if (
        p > 0.0
        and p < 1.0
    ):

        log_ratio_std[i] = (
            sigma_p
            / (
                p
                * (
                    1.0
                    - p
                )
            )
        )


theory_log_ratio = np.log(
    theory_right
    / theory_left
)


two_state_approximation = (
    2.0
    * forces
    / kBT_used
)


plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    forces,
    simulation_log_ratio,
    yerr=log_ratio_std,
    marker="o",
    capsize=4,
    label="Simulación"
)


plt.plot(
    forces,
    theory_log_ratio,
    linestyle="--",
    label="Teoría canónica"
)


plt.plot(
    forces,
    two_state_approximation,
    linestyle=":",
    label=r"Aproximación $2F/k_BT$"
)


plt.axhline(
    0.0,
    linestyle="-.",
    alpha=0.7
)


plt.axvline(
    0.0,
    linestyle="-.",
    alpha=0.7
)


plt.xlabel(
    "Fuerza externa $F$"
)

plt.ylabel(
    r"$\ln\left(P_+/P_-\right)$"
)


plt.title(
    "Razón de ocupaciones frente a la fuerza externa\n"
    f"$k_BT={kBT_used:g}$"
)


plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figures_folder
    / "double_well_log_occupation_ratio.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print("===================================")
print("BARRIDO DE FUERZA COMPLETADO")
print("===================================")

print()

print(
    f"Tabla individual:"
)

print(
    individual_file
)

print()

print(
    f"Tabla resumen:"
)

print(
    summary_file
)

print()

print(
    "Figuras generadas:"
)

print(
    figures_folder
    / "double_well_occupation_vs_force.png"
)

print(
    figures_folder
    / "double_well_potential_vs_force.png"
)

print(
    figures_folder
    / "double_well_residence_time_vs_force.png"
)

print(
    figures_folder
    / "double_well_log_occupation_ratio.png"
)

print()

print(
    "IMPORTANTE: los tiempos de estancia "
    "se calculan a partir de los puntos "
    "guardados en el CSV."
)

print(
    "Si entre dos muestras ocurren varias "
    "transiciones, estas no pueden detectarse."
)