from pathlib import Path

import csv
import os
import secrets
import subprocess

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(
    __file__
).resolve().parents[1]


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


# ===================================
# Parámetros del estudio
# ===================================

algorithm = "GJF"

eta = 1.0

A = 1.0

kBT = 0.2

mass = 1.0

dt = 0.01


# Tiempo de producción.
total_time = 100000.0


# Antes de empezar a medir obligamos
# al sistema a evolucionar durante
# al menos este tiempo.
min_thermalization_time = 5000.0


# Resolución suficiente para estudiar
# también las transiciones.
max_saved_points = 100000


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


initial_positions = [
    -1.0,
    0.0,
    1.0
]


# Más fuerzas para visualizar
# claramente la inclinación.
potential_plot_forces = [
    -0.4,
    -0.2,
    0.0,
    0.2,
    0.4
]


algorithms = {
    "EULER_MARUYAMA":
        "euler",

    "RUNGE_KUTTA":
        "rk",

    "GJF":
        "gjf"
}


algorithm_name = (
    algorithms[
        algorithm
    ]
)


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
        "Compila primero."
    )


data_file = (
    data_folder
    / f"harmonic_{algorithm_name}.csv"
)


# ===================================
# Funciones
# ===================================

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


def integrate(
    y,
    x
):

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
        - force
        * x
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
            weights[
                left_mask
            ],
            x[
                left_mask
            ]
        )
        / total
    )


    probability_right = (
        integrate(
            weights[
                right_mask
            ],
            x[
                right_mask
            ]
        )
        / total
    )


    return (
        probability_left,
        probability_right
    )


def occupation_statistics(
    positions
):

    right = float(
        np.mean(
            positions >= 0.0
        )
    )


    left = (
        1.0
        - right
    )


    transitions = int(
        np.count_nonzero(
            (
                positions[
                    1:
                ]
                >= 0.0
            )
            != (
                positions[
                    :-1
                ]
                >= 0.0
            )
        )
    )


    return (
        left,
        right,
        transitions
    )


def mean_and_std(
    values
):

    values = np.asarray(
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


def calculate_r_squared(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )


    predicted = np.asarray(
        predicted,
        dtype=float
    )


    ss_res = float(
        np.sum(
            (
                observed
                - predicted
            ) ** 2
        )
    )


    ss_tot = float(
        np.sum(
            (
                observed
                - np.mean(
                    observed
                )
            ) ** 2
        )
    )


    if ss_tot == 0.0:

        return np.nan


    return (
        1.0
        - ss_res
        / ss_tot
    )


# ===================================
# Simulaciones
# ===================================

original_config = (
    config_file.read_text()
)


base_seed = secrets.randbelow(
    (2 ** 32 - 1)
    - 1000000
)


results = []


try:

    for force_index, force_value in enumerate(
        force_values
    ):

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


        for (
            initial_index,
            x0
        ) in enumerate(
            initial_positions
        ):

            seed = (
                base_seed
                + force_index
                * 50000
                + initial_index
                * 11111
            )


            config_file.write_text(
                "model=DOUBLE_WELL\n"
                f"algorithm={algorithm}\n"
                f"eta={eta:.12g}\n"
                f"kBT={kBT:.12g}\n"
                f"mass={mass:.12g}\n"
                f"A={A:.12g}\n"
                f"force={force_value:.12g}\n"
                f"h={dt:.12g}\n"
                f"totalTime={total_time:.12g}\n"
                f"minThermalizationTime="
                f"{min_thermalization_time:.12g}\n"
                f"maxSavedPoints="
                f"{max_saved_points}\n"
                f"seed={seed}\n"
                f"x0={x0:.12g}\n"
                "v0=0.0\n"
            )


            print(
                f"x0={x0:g}, "
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


            positions = (
                data[
                    :,
                    1
                ]
            )


            (
                occupation_left,
                occupation_right,
                transitions
            ) = occupation_statistics(
                positions
            )


            (
                theory_left,
                theory_right
            ) = theoretical_occupancy(
                A,
                force_value,
                kBT
            )


            thermalization_time = float(
                parameters.get(
                    "thermalizationTime",
                    np.nan
                )
            )


            results.append({
                "force":
                    force_value,

                "x0":
                    x0,

                "seed":
                    seed,

                "occupation_left":
                    occupation_left,

                "occupation_right":
                    occupation_right,

                "theory_left":
                    theory_left,

                "theory_right":
                    theory_right,

                "transitions":
                    transitions,

                "thermalization_time":
                    thermalization_time
            })


            print(
                f"  t_therm = "
                f"{thermalization_time:.1f}"
            )


            print(
                f"  P(-) = "
                f"{occupation_left:.4f}"
            )


            print(
                f"  P(+) = "
                f"{occupation_right:.4f}"
            )


finally:

    # Recuperamos la configuración
    # que tuviera el usuario.
    config_file.write_text(
        original_config
    )


# ===================================
# Promedio de las realizaciones
# ===================================

summary = []


for force_value in force_values:

    rows = [
        row
        for row in results
        if np.isclose(
            row[
                "force"
            ],
            force_value
        )
    ]


    (
        left_mean,
        left_std
    ) = mean_and_std([
        row[
            "occupation_left"
        ]
        for row in rows
    ])


    (
        right_mean,
        right_std
    ) = mean_and_std([
        row[
            "occupation_right"
        ]
        for row in rows
    ])


    (
        transition_mean,
        transition_std
    ) = mean_and_std([
        row[
            "transitions"
        ]
        for row in rows
    ])


    (
        thermalization_mean,
        thermalization_std
    ) = mean_and_std([
        row[
            "thermalization_time"
        ]
        for row in rows
    ])


    summary.append({
        "force":
            force_value,

        "occupation_left_mean":
            left_mean,

        "occupation_left_std":
            left_std,

        "occupation_right_mean":
            right_mean,

        "occupation_right_std":
            right_std,

        "theory_left":
            rows[
                0
            ][
                "theory_left"
            ],

        "theory_right":
            rows[
                0
            ][
                "theory_right"
            ],

        "transitions_mean":
            transition_mean,

        "transitions_std":
            transition_std,

        "thermalization_time_mean":
            thermalization_mean,

        "thermalization_time_std":
            thermalization_std
    })


# ===================================
# Tablas
# ===================================

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
            results[
                0
            ].keys()
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
            summary[
                0
            ].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        summary
    )


# ===================================
# Arrays
# ===================================

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
# Ajuste de los puntos de simulación
# ===================================

valid_fit = (
    (right_mean > 0.0)
    & (right_mean < 1.0)
    & (left_mean > 0.0)
    & (left_mean < 1.0)
)


fit_forces = (
    forces[
        valid_fit
    ]
)


fit_log_ratio = np.log(
    right_mean[
        valid_fit
    ]
    / left_mean[
        valid_fit
    ]
)


fit_log_ratio_std = (
    right_std[
        valid_fit
    ]
    / (
        right_mean[
            valid_fit
        ]
        * left_mean[
            valid_fit
        ]
    )
)


weights = np.ones(
    len(
        fit_forces
    )
)


good_errors = (
    np.isfinite(
        fit_log_ratio_std
    )
    & (
        fit_log_ratio_std > 0.0
    )
)


weights[
    good_errors
] = (
    1.0
    / fit_log_ratio_std[
        good_errors
    ]
)


fit_coefficients = np.polyfit(
    fit_forces,
    fit_log_ratio,
    1,
    w=weights
)


fit_slope = float(
    fit_coefficients[
        0
    ]
)


fit_intercept = float(
    fit_coefficients[
        1
    ]
)


fit_prediction_points = (
    fit_slope
    * fit_forces
    + fit_intercept
)


fit_r_squared = (
    calculate_r_squared(
        fit_log_ratio,
        fit_prediction_points
    )
)


expected_slope = (
    2.0
    / kBT
)


slope_error_percent = (
    abs(
        fit_slope
        - expected_slope
    )
    / expected_slope
    * 100.0
)


if fit_slope > 0.0:

    effective_kBT = (
        2.0
        / fit_slope
    )

else:

    effective_kBT = np.nan


fit_file = (
    tables_folder
    / "double_well_force_fit.csv"
)


with fit_file.open(
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "parameter",
        "value"
    ])

    writer.writerow([
        "fit_slope",
        fit_slope
    ])

    writer.writerow([
        "fit_intercept",
        fit_intercept
    ])

    writer.writerow([
        "fit_R2",
        fit_r_squared
    ])

    writer.writerow([
        "expected_slope_2_over_kBT",
        expected_slope
    ])

    writer.writerow([
        "slope_relative_error_percent",
        slope_error_percent
    ])

    writer.writerow([
        "effective_kBT_from_fit",
        effective_kBT
    ])


# ===================================
# Curvas continuas
# ===================================

force_dense = np.linspace(
    force_values[
        0
    ],
    force_values[
        -1
    ],
    401
)


canonical_right_dense = []

canonical_left_dense = []


for force_value in force_dense:

    (
        p_left,
        p_right
    ) = theoretical_occupancy(
        A,
        force_value,
        kBT
    )


    canonical_left_dense.append(
        p_left
    )


    canonical_right_dense.append(
        p_right
    )


canonical_left_dense = np.array(
    canonical_left_dense
)


canonical_right_dense = np.array(
    canonical_right_dense
)


# Ajuste obtenido de la simulación:
#
# ln(P+/P-) = m F + b
#
# P+ = 1 / (1 + exp[-(mF+b)])
fit_right_dense = (
    1.0
    / (
        1.0
        + np.exp(
            -(
                fit_slope
                * force_dense
                + fit_intercept
            )
        )
    )
)


fit_left_dense = (
    1.0
    - fit_right_dense
)


# Aproximación de dos estados
# esperada teóricamente.
two_state_right_dense = (
    1.0
    / (
        1.0
        + np.exp(
            -2.0
            * force_dense
            / kBT
        )
    )
)


# ===================================
# 1. Ocupación de ambos pozos
# ===================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    force_dense,
    canonical_right_dense,
    linestyle="--",
    label=r"Teoría canónica $P_+$"
)


plt.plot(
    force_dense,
    canonical_left_dense,
    linestyle="--",
    label=r"Teoría canónica $P_-$"
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
    linestyle="none",
    label=r"Simulación $P_+$"
)


plt.errorbar(
    forces,
    left_mean,
    yerr=left_std,
    marker="o",
    capsize=4,
    linestyle="none",
    label=r"Simulación $P_-$"
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
    "Ocupación de los pozos "
    "frente a la fuerza externa\n"
    f"$A={A:g}$, "
    f"$\\eta={eta:g}$, "
    f"$k_BT={kBT:g}$"
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
# 2. Comprobación explícita del fit
# ===================================

plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    forces,
    right_mean,
    yerr=right_std,
    marker="o",
    capsize=4,
    linestyle="none",
    label=r"Simulación $P_+$"
)


plt.plot(
    force_dense,
    canonical_right_dense,
    linestyle="--",
    label="Teoría canónica"
)


plt.plot(
    force_dense,
    fit_right_dense,
    label=(
        "Fit simulación: "
        fr"$m={fit_slope:.3f}$"
    )
)


plt.plot(
    force_dense,
    two_state_right_dense,
    linestyle=":",
    label=(
        "Dos estados: "
        fr"$m=2/k_BT={expected_slope:.3f}$"
    )
)


plt.axhline(
    0.5,
    linestyle="-.",
    alpha=0.6
)


plt.axvline(
    0.0,
    linestyle="-.",
    alpha=0.6
)


plt.xlabel(
    "Fuerza externa $F$"
)


plt.ylabel(
    r"$P_+=P(x>0)$"
)


plt.ylim(
    -0.02,
    1.02
)


plt.title(
    "Ajuste de la ocupación "
    "del pozo derecho\n"
    fr"$R^2={fit_r_squared:.4f}$, "
    fr"$b={fit_intercept:.3f}$"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "double_well_occupation_fit.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 3. Potencial con más fuerzas
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
        A
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
    "Inclinación del doble pozo "
    "por una fuerza externa\n"
    f"$A={A:g}$"
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
# 4. Logaritmo del cociente
# ===================================

simulation_log_ratio = np.full(
    len(
        forces
    ),
    np.nan
)


log_ratio_std = np.full(
    len(
        forces
    ),
    np.nan
)


valid_log = (
    (right_mean > 0.0)
    & (right_mean < 1.0)
)


simulation_log_ratio[
    valid_log
] = np.log(
    right_mean[
        valid_log
    ]
    / (
        1.0
        - right_mean[
            valid_log
        ]
    )
)


log_ratio_std[
    valid_log
] = (
    right_std[
        valid_log
    ]
    / (
        right_mean[
            valid_log
        ]
        * (
            1.0
            - right_mean[
                valid_log
            ]
        )
    )
)


canonical_log_ratio_dense = np.log(
    canonical_right_dense
    / canonical_left_dense
)


two_state_log_ratio_dense = (
    expected_slope
    * force_dense
)


fit_log_ratio_dense = (
    fit_slope
    * force_dense
    + fit_intercept
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
    linestyle="none",
    label="Simulación"
)


plt.plot(
    force_dense,
    canonical_log_ratio_dense,
    linestyle="--",
    label="Teoría canónica"
)


plt.plot(
    force_dense,
    fit_log_ratio_dense,
    label=(
        "Ajuste simulación: "
        fr"$m={fit_slope:.3f}$, "
        fr"$R^2={fit_r_squared:.4f}$"
    )
)


plt.plot(
    force_dense,
    two_state_log_ratio_dense,
    linestyle=":",
    label=(
        r"Dos estados: "
        fr"$m=2/k_BT={expected_slope:.3f}$"
    )
)


plt.axhline(
    0.0,
    linestyle="-.",
    alpha=0.6
)


plt.axvline(
    0.0,
    linestyle="-.",
    alpha=0.6
)


plt.xlabel(
    "Fuerza externa $F$"
)


plt.ylabel(
    r"$\ln(P_+/P_-)$"
)


plt.title(
    "Razón de ocupaciones "
    "frente a la fuerza externa"
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


# ===================================
# Resumen
# ===================================

print()

print(
    "==================================="
)

print(
    "BARRIDO DE FUERZA COMPLETADO"
)

print(
    "==================================="
)

print()

print(
    "Parámetros:"
)

print(
    f"  A = {A}"
)

print(
    f"  eta = {eta}"
)

print(
    f"  kBT = {kBT}"
)

print(
    f"  h = {dt}"
)

print(
    f"  T produccion = {total_time}"
)

print(
    "  T minimo termalizacion = "
    f"{min_thermalization_time}"
)

print()

print(
    "Ajuste:"
)

print(
    "  ln(P+/P-) = m F + b"
)

print(
    f"  m experimental = "
    f"{fit_slope:.5f}"
)

print(
    f"  m teorica = "
    f"{expected_slope:.5f}"
)

print(
    f"  error pendiente = "
    f"{slope_error_percent:.2f} %"
)

print(
    f"  b = "
    f"{fit_intercept:.5f}"
)

print(
    f"  R2 = "
    f"{fit_r_squared:.5f}"
)

print(
    f"  kBT efectivo = "
    f"{effective_kBT:.5f}"
)

print()

print(
    "Tablas:"
)

print(
    individual_file
)

print(
    summary_file
)

print(
    fit_file
)

print()

print(
    "Figuras:"
)

print(
    figures_folder
    / "double_well_occupation_vs_force.png"
)

print(
    figures_folder
    / "double_well_occupation_fit.png"
)

print(
    figures_folder
    / "double_well_potential_vs_force.png"
)

print(
    figures_folder
    / "double_well_log_occupation_ratio.png"
)