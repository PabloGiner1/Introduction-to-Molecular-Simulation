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


data_file = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
    / "harmonic_gjf.csv"
)


parameters_file = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
    / "parameters.txt"
)


results_folder = (
    project_root
    / "results"
    / "double_well"
    / "residence_time"
)


figures_folder = (
    results_folder
    / "figures"
)


tables_folder = (
    results_folder
    / "tables"
)


figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


tables_folder.mkdir(
    parents=True,
    exist_ok=True
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


# ===================================
# Parámetros del estudio
# ===================================

algorithm = "GJF"

eta = 1.0

kBT = 0.2

mass = 1.0

dt = 0.01

total_time = 30000.0


# Número máximo de puntos
# almacenados en el CSV.
max_saved_points = 120000


# Alturas de la barrera.
A_values = np.array([
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
])


# Tres realizaciones independientes.
initial_positions = [
    -1.0,
    0.0,
    1.0
]


# Consideramos una transición completa
# cuando se alcanza el umbral opuesto.
threshold = 0.5


# ===================================
# Funciones auxiliares
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


def extract_residence_times(
    time,
    positions,
    c
):

    # Estados bien definidos:
    #
    # pozo izquierdo: x <= -c
    # pozo derecho:   x >= +c
    #
    # La región -c < x < c se considera
    # parte de una transición y no crea
    # por sí sola un nuevo estado.

    residence_times = []

    residence_states = []


    current_state = 0

    start_time = 0.0

    start_index = None


    # Buscamos el primer estado
    # claramente definido.
    for i in range(
        len(
            positions
        )
    ):

        if (
            positions[
                i
            ]
            <= -c
        ):

            current_state = -1

            start_time = (
                time[
                    i
                ]
            )

            start_index = i

            break


        if (
            positions[
                i
            ]
            >= c
        ):

            current_state = 1

            start_time = (
                time[
                    i
                ]
            )

            start_index = i

            break


    if start_index is None:

        return (
            np.array(
                [],
                dtype=float
            ),
            np.array(
                [],
                dtype=int
            )
        )


    # El primer intervalo puede haber
    # comenzado antes del inicio del CSV.
    first_interval = True


    for i in range(
        start_index + 1,
        len(
            positions
        )
    ):

        transition = False


        if (
            current_state == -1
            and positions[
                i
            ]
            >= c
        ):

            transition = True

            new_state = 1


        elif (
            current_state == 1
            and positions[
                i
            ]
            <= -c
        ):

            transition = True

            new_state = -1


        if transition:

            duration = (
                time[
                    i
                ]
                - start_time
            )


            if not first_interval:

                if duration > 0.0:

                    residence_times.append(
                        duration
                    )

                    residence_states.append(
                        current_state
                    )

            else:

                first_interval = False


            current_state = (
                new_state
            )


            start_time = (
                time[
                    i
                ]
            )


    # El último intervalo tampoco
    # se utiliza, porque puede quedar
    # truncado al terminar la simulación.

    return (
        np.array(
            residence_times,
            dtype=float
        ),
        np.array(
            residence_states,
            dtype=int
        )
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


replicate_results = []

summary_results = []

all_residence_rows = []


try:

    for A_index, A in enumerate(
        A_values
    ):

        print()

        print(
            "==================================="
        )

        print(
            f"A = {A:g}"
        )

        print(
            "==================================="
        )


        pooled_residence_times = []

        replicate_means = []


        for (
            replicate_index,
            x0
        ) in enumerate(
            initial_positions
        ):

            seed = (
                base_seed
                + A_index
                * 100000
                + replicate_index
                * 11111
            )


            config_file.write_text(
                "model=DOUBLE_WELL\n"
                f"algorithm={algorithm}\n"
                f"eta={eta:.12g}\n"
                f"kBT={kBT:.12g}\n"
                f"mass={mass:.12g}\n"
                f"A={A:.12g}\n"
                "force=0.0\n"
                f"h={dt:.12g}\n"
                f"totalTime={total_time:.12g}\n"
                f"maxSavedPoints={max_saved_points}\n"
                f"seed={seed}\n"
                f"x0={x0:.12g}\n"
                "v0=0.0\n"
            )


            print(
                f"Rep {replicate_index + 1}: "
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
                    "Fallo de simulacion "
                    f"para A={A:g}\n"
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
                    f"A={A:g}"
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


            (
                residence_times,
                residence_states
            ) = extract_residence_times(
                time,
                positions,
                threshold
            )


            if len(
                residence_times
            ) > 0:

                replicate_mean = float(
                    np.mean(
                        residence_times
                    )
                )


                replicate_means.append(
                    replicate_mean
                )


                pooled_residence_times.extend(
                    residence_times.tolist()
                )

            else:

                replicate_mean = np.nan


            for (
                tau,
                state
            ) in zip(
                residence_times,
                residence_states
            ):

                all_residence_rows.append({
                    "A":
                        A,

                    "replicate":
                        replicate_index + 1,

                    "x0":
                        x0,

                    "seed":
                        seed,

                    "well":
                        "right"
                        if state == 1
                        else "left",

                    "tau":
                        tau
                })


            replicate_results.append({
                "A":
                    A,

                "replicate":
                    replicate_index + 1,

                "x0":
                    x0,

                "seed":
                    seed,

                "residence_count":
                    len(
                        residence_times
                    ),

                "mu":
                    replicate_mean,

                "thermalization_time":
                    float(
                        parameters.get(
                            "thermalizationTime",
                            np.nan
                        )
                    ),

                "saved_dt":
                    float(
                        parameters.get(
                            "savedDt",
                            np.nan
                        )
                    )
            })


            print(
                f"  N estancias = "
                f"{len(residence_times)}"
            )


            if np.isfinite(
                replicate_mean
            ):

                print(
                    f"  mu = "
                    f"{replicate_mean:.3f}"
                )

            else:

                print(
                    "  mu = nan"
                )


        pooled_residence_times = np.array(
            pooled_residence_times,
            dtype=float
        )


        (
            mu_replicates,
            mu_replicates_std
        ) = mean_and_std(
            replicate_means
        )


        if len(
            pooled_residence_times
        ) > 0:

            mu_pooled = float(
                np.mean(
                    pooled_residence_times
                )
            )

        else:

            mu_pooled = np.nan


        summary_results.append({
            "A":
                A,

            "residence_count":
                len(
                    pooled_residence_times
                ),

            "mu":
                mu_replicates,

            "mu_std":
                mu_replicates_std,

            "mu_pooled":
                mu_pooled
        })


        # ===================================
        # Distribución de tiempos
        # ===================================

        if (
            len(
                pooled_residence_times
            ) > 0
            and np.isfinite(
                mu_pooled
            )
        ):

            tau_max_plot = float(
                np.percentile(
                    pooled_residence_times,
                    99
                )
            )


            tau_grid = np.linspace(
                0.0,
                tau_max_plot,
                500
            )


            theoretical_distribution = (
                1.0
                / mu_pooled
                * np.exp(
                    -tau_grid
                    / mu_pooled
                )
            )


            plt.figure(
                figsize=(8, 5)
            )


            plt.hist(
                pooled_residence_times,
                bins="auto",
                density=True,
                alpha=0.55,
                label="Simulación"
            )


            plt.plot(
                tau_grid,
                theoretical_distribution,
                linestyle="--",
                linewidth=2,
                label=(
                    r"$p(\tau)=\mu^{-1}"
                    r"e^{-\tau/\mu}$"
                    f", $\\mu={mu_pooled:.2f}$"
                )
            )


            plt.xlabel(
                r"Tiempo de estancia $\tau$"
            )


            plt.ylabel(
                "Densidad de probabilidad"
            )


            plt.title(
                "Distribución de tiempos "
                "de estancia\n"
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
                / (
                    "residence_distribution"
                    f"_A_{A:.1f}.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )


            plt.close()


finally:

    # Recuperamos siempre
    # simulation.txt original.
    config_file.write_text(
        original_config
    )


# ===================================
# Guardar tablas
# ===================================

replicate_file = (
    tables_folder
    / "residence_replicates.csv"
)


with replicate_file.open(
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            replicate_results[
                0
            ].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        replicate_results
    )


summary_file = (
    tables_folder
    / "residence_summary.csv"
)


with summary_file.open(
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            summary_results[
                0
            ].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        summary_results
    )


residence_file = (
    tables_folder
    / "residence_times.csv"
)


with residence_file.open(
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "A",
            "replicate",
            "x0",
            "seed",
            "well",
            "tau"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_residence_rows
    )


# ===================================
# Tiempo medio frente a A
# ===================================

A_plot = np.array([
    row[
        "A"
    ]
    for row in summary_results
])


mu_plot = np.array([
    row[
        "mu"
    ]
    for row in summary_results
])


mu_std_plot = np.array([
    row[
        "mu_std"
    ]
    for row in summary_results
])


plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    A_plot,
    mu_plot,
    yerr=mu_std_plot,
    marker="o",
    capsize=4,
    linestyle="-",
    label="Simulación"
)


plt.yscale(
    "log"
)


plt.xlabel(
    "Altura de barrera $A$"
)


plt.ylabel(
    r"Tiempo medio "
    r"$\mu=\langle\tau\rangle$"
)


plt.title(
    "Tiempo medio de estancia "
    "frente a la barrera"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "mean_residence_vs_A.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# Distribución normalizada tau / mu
# ===================================

plt.figure(
    figsize=(8, 5)
)


normalized_bins = np.linspace(
    0.0,
    8.0,
    65
)


for row in summary_results:

    A = row[
        "A"
    ]


    mu = row[
        "mu_pooled"
    ]


    if not np.isfinite(
        mu
    ):

        continue


    tau_values = np.array([
        residence_row[
            "tau"
        ]
        for residence_row
        in all_residence_rows
        if np.isclose(
            residence_row[
                "A"
            ],
            A
        )
    ])


    if len(
        tau_values
    ) == 0:

        continue


    normalized_tau = (
        tau_values
        / mu
    )


    plt.hist(
        normalized_tau,
        bins=normalized_bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=fr"$A={A:g}$"
    )


u = np.linspace(
    0.0,
    8.0,
    500
)


plt.plot(
    u,
    np.exp(
        -u
    ),
    linestyle="--",
    linewidth=2,
    label=r"$e^{-\tau/\mu}$"
)


plt.xlabel(
    r"Tiempo normalizado "
    r"$\tau/\mu$"
)


plt.ylabel(
    "Densidad de probabilidad"
)


plt.title(
    "Distribución normalizada "
    "de tiempos de estancia"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "normalized_residence_times.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# Ajuste ln(mu) frente a A
# ===================================

valid_mask = (
    np.isfinite(
        mu_plot
    )
    & (
        mu_plot
        > 0.0
    )
)


A_fit = (
    A_plot[
        valid_mask
    ]
)


mu_fit = (
    mu_plot[
        valid_mask
    ]
)


mu_std_fit = (
    mu_std_plot[
        valid_mask
    ]
)


log_mu = np.log(
    mu_fit
)


linear_fit = np.polyfit(
    A_fit,
    log_mu,
    1
)


slope = float(
    linear_fit[
        0
    ]
)


intercept = float(
    linear_fit[
        1
    ]
)


prediction = (
    slope
    * A_fit
    + intercept
)


ss_res = float(
    np.sum(
        (
            log_mu
            - prediction
        ) ** 2
    )
)


ss_tot = float(
    np.sum(
        (
            log_mu
            - np.mean(
                log_mu
            )
        ) ** 2
    )
)


r_squared = (
    1.0
    - ss_res
    / ss_tot
    if ss_tot > 0.0
    else np.nan
)


A_line = np.linspace(
    np.min(
        A_fit
    ),
    np.max(
        A_fit
    ),
    300
)


plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    A_fit,
    log_mu,
    yerr=(
        mu_std_fit
        / mu_fit
    ),
    marker="o",
    linestyle="none",
    capsize=4,
    label="Simulación"
)


plt.plot(
    A_line,
    slope
    * A_line
    + intercept,
    label=(
        "Ajuste: "
        fr"$m={slope:.3f}$, "
        fr"$R^2={r_squared:.4f}$"
    )
)


plt.xlabel(
    "Altura de barrera $A$"
)


plt.ylabel(
    r"$\ln\mu$"
)


plt.title(
    "Ley exponencial del "
    "tiempo de residencia\n"
    fr"$1/k_BT={1.0/kBT:.3f}$"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "log_mean_residence_vs_A.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# Linealización de Kramers
# ln(A mu) frente a A
# ===================================

kramers_y = np.log(
    A_fit
    * mu_fit
)


kramers_fit = np.polyfit(
    A_fit,
    kramers_y,
    1
)


kramers_slope = float(
    kramers_fit[
        0
    ]
)


kramers_intercept = float(
    kramers_fit[
        1
    ]
)


kramers_prediction = (
    kramers_slope
    * A_fit
    + kramers_intercept
)


ss_res_kramers = float(
    np.sum(
        (
            kramers_y
            - kramers_prediction
        ) ** 2
    )
)


ss_tot_kramers = float(
    np.sum(
        (
            kramers_y
            - np.mean(
                kramers_y
            )
        ) ** 2
    )
)


r_squared_kramers = (
    1.0
    - ss_res_kramers
    / ss_tot_kramers
    if ss_tot_kramers > 0.0
    else np.nan
)


theory_slope = (
    1.0
    / kBT
)


theory_intercept = float(
    np.mean(
        kramers_y
        - theory_slope
        * A_fit
    )
)


plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    A_fit,
    kramers_y,
    yerr=(
        mu_std_fit
        / mu_fit
    ),
    marker="o",
    linestyle="none",
    capsize=4,
    label="Simulación"
)


plt.plot(
    A_line,
    kramers_slope
    * A_line
    + kramers_intercept,
    label=(
        "Ajuste: "
        fr"$m={kramers_slope:.3f}$, "
        fr"$R^2={r_squared_kramers:.4f}$"
    )
)


plt.plot(
    A_line,
    theory_slope
    * A_line
    + theory_intercept,
    linestyle="--",
    label=(
        "Pendiente esperada "
        fr"$1/k_BT={theory_slope:.3f}$"
    )
)


plt.xlabel(
    "Altura de barrera $A$"
)


plt.ylabel(
    r"$\ln(A\mu)$"
)


plt.title(
    "Linealización de la "
    "ley de Kramers"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "kramers_linearization.png",
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
    "TIEMPOS DE RESIDENCIA"
)

print(
    "==================================="
)

print()

print(
    "Método utilizado:"
)

print(
    "Transiciones entre dos umbrales"
)

print(
    f"x = -{threshold:g} "
    f"y x = +{threshold:g}"
)

print()


for row in summary_results:

    print(
        f"A = {row['A']:.1f}"
    )

    print(
        f"  N estancias = "
        f"{row['residence_count']}"
    )

    print(
        f"  mu = "
        f"{row['mu_pooled']:.3f}"
    )

    print()


print(
    "Ajuste ln(mu) vs A:"
)

print(
    f"  pendiente = "
    f"{slope:.4f}"
)

print(
    f"  R2 = "
    f"{r_squared:.4f}"
)

print()


print(
    "Linealización de Kramers:"
)

print(
    f"  pendiente simulada = "
    f"{kramers_slope:.4f}"
)

print(
    f"  pendiente teorica = "
    f"{theory_slope:.4f}"
)

print(
    f"  R2 = "
    f"{r_squared_kramers:.4f}"
)

print()

print(
    "Resultados guardados en:"
)

print(
    results_folder
)