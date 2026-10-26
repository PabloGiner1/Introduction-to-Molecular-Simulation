from pathlib import Path

import csv
import os
import secrets
import shutil
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
    / "polymer"
    / "polymer.csv"
)


parameters_file = (
    project_root
    / "data"
    / "raw"
    / "polymer"
    / "parameters.txt"
)


results_folder = (
    project_root
    / "results"
    / "polymer"
    / "size_sweep"
)


figures_folder = (
    results_folder
    / "figures"
)


tables_folder = (
    results_folder
    / "tables"
)


raw_folder = (
    project_root
    / "data"
    / "raw"
    / "polymer"
    / "size_sweep"
)


figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


tables_folder.mkdir(
    parents=True,
    exist_ok=True
)


raw_folder.mkdir(
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

N_values = [
    4,
    8,
    16,
    32,
    64
]


algorithm = "GJF"


ke = 100.0

l0 = 1.0


mass = 1.0

kBT = 1.0

eta = 1.0


dt = 0.005


# Antes usábamos 1000.
# Ahora aumentamos claramente
# el tiempo de producción.
total_time = 10000.0


# También obligamos al polímero
# a evolucionar bastante antes
# de comenzar a guardar medidas.
min_thermalization_time = 2000.0


max_saved_points = 10000


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


# ===================================
# Valores teóricos de un enlace
# ===================================

l_grid = np.linspace(
    0.0,
    l0
    + 10.0
    * np.sqrt(
        kBT
        / ke
    ),
    50000
)


bond_potential = (
    0.5
    * ke
    * (
        l_grid
        - l0
    ) ** 2
)


bond_weight = (
    l_grid ** 2
    * np.exp(
        -bond_potential
        / kBT
    )
)


bond_normalization = integrate(
    bond_weight,
    l_grid
)


bond_length_theory = float(
    integrate(
        l_grid
        * bond_weight,
        l_grid
    )
    / bond_normalization
)


elastic_per_bond_theory = float(
    integrate(
        bond_potential
        * bond_weight,
        l_grid
    )
    / bond_normalization
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

    for index, N in enumerate(
        N_values
    ):

        seed = (
            base_seed
            + index
            * 100003
        )


        print()

        print(
            "==================================="
        )

        print(
            f"POLIMERO N = {N}"
        )

        print(
            "==================================="
        )


        config_file.write_text(
            "model=POLYMER\n"
            f"algorithm={algorithm}\n"
            f"N={N}\n"
            f"ke={ke:.12g}\n"
            f"l0={l0:.12g}\n"
            f"mass={mass:.12g}\n"
            f"kBT={kBT:.12g}\n"
            f"eta={eta:.12g}\n"
            f"h={dt:.12g}\n"
            f"totalTime={total_time:.12g}\n"
            f"minThermalizationTime="
            f"{min_thermalization_time:.12g}\n"
            f"maxSavedPoints="
            f"{max_saved_points}\n"
            f"seed={seed}\n"
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
                f"para N={N}\n"
                f"{simulation_result.stdout}\n"
                f"{simulation_result.stderr}"
            )


        if not data_file.exists():

            raise RuntimeError(
                "No se ha generado "
                "polymer.csv."
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
                "Datos no validos "
                f"para N={N}"
            )


        time = (
            data[
                :,
                0
            ]
        )


        kinetic = (
            data[
                :,
                1
            ]
        )


        elastic = (
            data[
                :,
                2
            ]
        )


        coordinates = (
            data[
                :,
                3:
            ]
            .reshape(
                len(data),
                N,
                3
            )
        )


        # ==========================
        # Energías
        # ==========================

        mean_kinetic = float(
            np.mean(
                kinetic
            )
        )


        mean_elastic = float(
            np.mean(
                elastic
            )
        )


        kinetic_theory = (
            1.5
            * N
            * kBT
        )


        elastic_theory = (
            (N - 1)
            * elastic_per_bond_theory
        )


        kinetic_per_particle = (
            mean_kinetic
            / N
        )


        elastic_per_particle = (
            mean_elastic
            / N
        )


        kinetic_per_particle_theory = (
            kinetic_theory
            / N
        )


        elastic_per_particle_theory = (
            elastic_theory
            / N
        )


        kinetic_error = (
            abs(
                mean_kinetic
                - kinetic_theory
            )
            / kinetic_theory
            * 100.0
        )


        elastic_error = (
            abs(
                mean_elastic
                - elastic_theory
            )
            / elastic_theory
            * 100.0
        )


        # ==========================
        # Longitudes de enlace
        # ==========================

        bond_vectors = (
            coordinates[
                :,
                1:,
                :
            ]
            - coordinates[
                :,
                :-1,
                :
            ]
        )


        bond_lengths = np.linalg.norm(
            bond_vectors,
            axis=2
        )


        mean_bond_length = float(
            np.mean(
                bond_lengths
            )
        )


        bond_length_std = float(
            np.std(
                bond_lengths
            )
        )


        # ==========================
        # Convergencia acumulada
        # ==========================

        sample_count = (
            np.arange(
                len(
                    time
                )
            )
            + 1
        )


        cumulative_kinetic = (
            np.cumsum(
                kinetic
            )
            / sample_count
            / N
        )


        cumulative_elastic = (
            np.cumsum(
                elastic
            )
            / sample_count
            / N
        )


        results.append({
            "N":
                N,

            "seed":
                seed,

            "mean_kinetic":
                mean_kinetic,

            "kinetic_theory":
                kinetic_theory,

            "kinetic_error_percent":
                kinetic_error,

            "kinetic_per_particle":
                kinetic_per_particle,

            "kinetic_per_particle_theory":
                kinetic_per_particle_theory,

            "mean_elastic":
                mean_elastic,

            "elastic_theory":
                elastic_theory,

            "elastic_error_percent":
                elastic_error,

            "elastic_per_particle":
                elastic_per_particle,

            "elastic_per_particle_theory":
                elastic_per_particle_theory,

            "mean_bond_length":
                mean_bond_length,

            "bond_length_theory":
                bond_length_theory,

            "bond_length_std":
                bond_length_std,

            "thermalization_time":
                float(
                    parameters[
                        "thermalizationTime"
                    ]
                ),

            "saved_dt":
                float(
                    parameters[
                        "savedDt"
                    ]
                ),

            "time":
                time,

            "cumulative_kinetic":
                cumulative_kinetic,

            "cumulative_elastic":
                cumulative_elastic
        })


        # Conservamos también los
        # datos brutos de cada N.
        case_folder = (
            raw_folder
            / f"N_{N}"
        )


        case_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        shutil.copy2(
            data_file,
            case_folder
            / "polymer.csv"
        )


        shutil.copy2(
            parameters_file,
            case_folder
            / "parameters.txt"
        )


        print(
            "Energia cinetica:"
        )

        print(
            f"  <Ecin>/N = "
            f"{kinetic_per_particle:.5f}"
        )

        print(
            f"  Teoria   = "
            f"{kinetic_per_particle_theory:.5f}"
        )


        print(
            "Energia elastica:"
        )

        print(
            f"  <Eel>/N = "
            f"{elastic_per_particle:.5f}"
        )

        print(
            f"  Teoria  = "
            f"{elastic_per_particle_theory:.5f}"
        )


        print(
            "Longitud de enlace:"
        )

        print(
            f"  <l> = "
            f"{mean_bond_length:.5f}"
        )

        print(
            f"  Teoria = "
            f"{bond_length_theory:.5f}"
        )


finally:

    # Recuperamos la configuración
    # original del usuario.
    config_file.write_text(
        original_config
    )


# ===================================
# Tabla resumen
# ===================================

summary_rows = []


for result in results:

    summary_rows.append({
        "N":
            result[
                "N"
            ],

        "seed":
            result[
                "seed"
            ],

        "mean_kinetic":
            result[
                "mean_kinetic"
            ],

        "kinetic_theory":
            result[
                "kinetic_theory"
            ],

        "kinetic_error_percent":
            result[
                "kinetic_error_percent"
            ],

        "kinetic_per_particle":
            result[
                "kinetic_per_particle"
            ],

        "kinetic_per_particle_theory":
            result[
                "kinetic_per_particle_theory"
            ],

        "mean_elastic":
            result[
                "mean_elastic"
            ],

        "elastic_theory":
            result[
                "elastic_theory"
            ],

        "elastic_error_percent":
            result[
                "elastic_error_percent"
            ],

        "elastic_per_particle":
            result[
                "elastic_per_particle"
            ],

        "elastic_per_particle_theory":
            result[
                "elastic_per_particle_theory"
            ],

        "mean_bond_length":
            result[
                "mean_bond_length"
            ],

        "bond_length_theory":
            result[
                "bond_length_theory"
            ],

        "bond_length_std":
            result[
                "bond_length_std"
            ],

        "thermalization_time":
            result[
                "thermalization_time"
            ],

        "saved_dt":
            result[
                "saved_dt"
            ]
    })


summary_file = (
    tables_folder
    / "polymer_size_summary.csv"
)


with summary_file.open(
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            summary_rows[
                0
            ].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ===================================
# 1. Convergencia cinética / N
# ===================================

plt.figure(
    figsize=(9, 6)
)


for result in results:

    plt.plot(
        result[
            "time"
        ],
        result[
            "cumulative_kinetic"
        ],
        label=(
            f"$N={result['N']}$: "
            f"exp={result['kinetic_per_particle']:.3f}"
        )
    )


plt.axhline(
    1.5 * kBT,
    linestyle="--",
    label=(
        "Teoría: "
        r"$\langle E_{cin}\rangle/N"
        r"=3k_BT/2"
        f"={1.5*kBT:.3f}$"
    )
)


plt.xlabel(
    "Tiempo de producción $t$"
)


plt.ylabel(
    r"$\langle E_{cin}\rangle_{acum}/N$"
)


plt.title(
    "Convergencia de la energía cinética "
    "por partícula"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_kinetic_per_particle_convergence.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 2. Convergencia elástica / N
# ===================================

plt.figure(
    figsize=(9, 6)
)


for result in results:

    plt.plot(
        result[
            "time"
        ],
        result[
            "cumulative_elastic"
        ],
        label=(
            f"$N={result['N']}$: "
            f"exp="
            f"{result['elastic_per_particle']:.3f}, "
            f"teo="
            f"{result['elastic_per_particle_theory']:.3f}"
        )
    )


plt.xlabel(
    "Tiempo de producción $t$"
)


plt.ylabel(
    r"$\langle E_{el}\rangle_{acum}/N$"
)


plt.title(
    "Convergencia de la energía elástica "
    "por partícula"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_elastic_per_particle_convergence.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# Arrays para gráficas frente a N
# ===================================

N_plot = np.array([
    result[
        "N"
    ]
    for result in results
])


kinetic_per_particle = np.array([
    result[
        "kinetic_per_particle"
    ]
    for result in results
])


kinetic_per_particle_theory = np.array([
    result[
        "kinetic_per_particle_theory"
    ]
    for result in results
])


elastic_per_particle = np.array([
    result[
        "elastic_per_particle"
    ]
    for result in results
])


elastic_per_particle_theory = np.array([
    result[
        "elastic_per_particle_theory"
    ]
    for result in results
])


bond_length_mean = np.array([
    result[
        "mean_bond_length"
    ]
    for result in results
])


# ===================================
# 3. Energías normalizadas vs N
# ===================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    N_plot,
    kinetic_per_particle,
    marker="o",
    label=r"Simulación $\langle E_{cin}\rangle/N$"
)


plt.plot(
    N_plot,
    kinetic_per_particle_theory,
    linestyle="--",
    label=r"Teoría $\langle E_{cin}\rangle/N$"
)


plt.plot(
    N_plot,
    elastic_per_particle,
    marker="s",
    label=r"Simulación $\langle E_{el}\rangle/N$"
)


plt.plot(
    N_plot,
    elastic_per_particle_theory,
    linestyle="--",
    label=r"Teoría $\langle E_{el}\rangle/N$"
)


plt.xlabel(
    "Número de partículas $N$"
)


plt.ylabel(
    "Energía media por partícula"
)


plt.title(
    "Energías normalizadas "
    "frente al tamaño del polímero\n"
    f"$k_e={ke:g}$, "
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
    / "polymer_energy_per_particle_vs_N.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 4. Longitud de enlace vs N
# ===================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    N_plot,
    bond_length_mean,
    marker="o",
    label=(
        r"Simulación "
        r"$\langle l\rangle$"
    )
)


plt.axhline(
    bond_length_theory,
    linestyle="--",
    label=(
        r"Teoría "
        r"$\langle l\rangle$"
        f" = {bond_length_theory:.4f}"
    )
)


plt.axhline(
    l0,
    linestyle=":",
    label=(
        f"$l_0={l0:g}$"
    )
)


plt.xlabel(
    "Número de partículas $N$"
)


plt.ylabel(
    r"Longitud media de enlace "
    r"$\langle l\rangle$"
)


plt.title(
    "Longitud media de enlace "
    "frente al tamaño del polímero"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_bond_length_vs_N.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# Resumen por terminal
# ===================================

print()

print(
    "==================================="
)

print(
    "BARRIDO DE TAMANO DEL POLIMERO"
)

print(
    "==================================="
)

print()

print(
    f"ke = {ke}"
)

print(
    f"l0 = {l0}"
)

print(
    f"kBT = {kBT}"
)

print(
    f"eta = {eta}"
)

print(
    f"h = {dt}"
)

print(
    f"T produccion = {total_time}"
)

print(
    "T minimo termalizacion = "
    f"{min_thermalization_time}"
)

print()

print(
    "Valor teorico cinetico "
    "por particula:"
)

print(
    f"  {1.5*kBT:.6f}"
)

print()

print(
    "Valor teorico longitud "
    "de enlace:"
)

print(
    f"  {bond_length_theory:.6f}"
)

print()

print(
    "Resultados:"
)

for result in results:

    print()

    print(
        f"N = {result['N']}"
    )

    print(
        "  <Ecin>/N:"
    )

    print(
        f"    exp = "
        f"{result['kinetic_per_particle']:.6f}"
    )

    print(
        f"    teo = "
        f"{result['kinetic_per_particle_theory']:.6f}"
    )

    print(
        "  <Eel>/N:"
    )

    print(
        f"    exp = "
        f"{result['elastic_per_particle']:.6f}"
    )

    print(
        f"    teo = "
        f"{result['elastic_per_particle_theory']:.6f}"
    )

    print(
        "  <l>:"
    )

    print(
        f"    exp = "
        f"{result['mean_bond_length']:.6f}"
    )

    print(
        f"    teo = "
        f"{bond_length_theory:.6f}"
    )

    print(
        "  t thermalization = "
        f"{result['thermalization_time']:.3f}"
    )


print()

print(
    "Tabla:"
)

print(
    summary_file
)

print()

print(
    "Figuras:"
)

print(
    figures_folder
)