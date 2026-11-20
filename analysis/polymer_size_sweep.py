from pathlib import Path

import csv
import secrets
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from polymer_metrics import compute_time_averaged_msd


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


simulation_program = (
    project_root
    / "bin"
    / "simulation.exe"
)

if not simulation_program.exists():

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


total_time = 10000.0


min_thermalization_time = 2000.0


# Se descarta esta parte inicial de la producción para
# permitir que se relaje la conformación de la cadena.
production_burn_in_time = 2000.0


max_saved_points = 10000

number_of_blocks = 10


if not 0.0 <= production_burn_in_time < total_time:

    raise ValueError(
        "El tiempo de descarte debe ser no negativo "
        "y menor que el tiempo de produccion."
    )


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


def integrate(y, x):

    try:
        return np.trapezoid(
            y,
            x
        )

    except AttributeError:
        return np.trapz(
            y,
            x
        )


def block_standard_error(
    values,
    requested_blocks=10
):

    values = np.asarray(
        values,
        dtype=float
    )


    n_samples = len(
        values
    )


    if n_samples < 4:

        return (
            np.nan,
            1,
            n_samples
        )


    actual_blocks = min(
        requested_blocks,
        max(
            2,
            n_samples // 50
        )
    )


    block_size = (
        n_samples
        // actual_blocks
    )


    usable_samples = (
        block_size
        * actual_blocks
    )


    trimmed_values = (
        values[
            :usable_samples
        ]
    )


    block_means = (
        trimmed_values
        .reshape(
            actual_blocks,
            block_size
        )
        .mean(
            axis=1
        )
    )


    standard_error = float(
        np.std(
            block_means,
            ddof=1
        )
        / np.sqrt(
            actual_blocks
        )
    )


    return (
        standard_error,
        actual_blocks,
        block_size
    )


# ===================================
# Valores teóricos de un enlace
# ===================================

sigma = np.sqrt(
    kBT
    / ke
)


l_grid = np.linspace(
    0.0,
    l0
    + 10.0
    * sigma,
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


bond_probability = (
    bond_weight
    / bond_normalization
)


bond_length_theory = float(
    integrate(
        l_grid
        * bond_probability,
        l_grid
    )
)


bond_length_sq_theory = float(
    integrate(
        (l_grid ** 2)
        * bond_weight,
        l_grid
    )
    / bond_normalization
)


elastic_per_bond_theory = float(
    integrate(
        bond_potential
        * bond_probability,
        l_grid
    )
)


# ===================================
# Simulaciones
# ===================================

original_config = (
    config_file.read_text()
)

original_config_values = parse_key_values(
    original_config
)
first_end = original_config_values.get(
    "firstEnd"
)

if first_end not in {"fixed", "free"}:
    raise RuntimeError(
        "El parámetro firstEnd en config/simulation.txt "
        "debe ser 'fixed' o 'free'."
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
            f"firstEnd={first_end}\n"
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


        with data_file.open(
            newline=""
        ) as file:

            column_names = next(
                csv.reader(
                    file
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
            len(
                data
            ) < 2
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

        equilibrium_mask = (
            time >= production_burn_in_time
        )


        equilibrium_sample_count = int(
            np.count_nonzero(
                equilibrium_mask
            )
        )


        if equilibrium_sample_count < 2:

            raise RuntimeError(
                "No hay suficientes muestras después del "
                f"descarte inicial para N={N}. "
                f"Tiempo de descarte={production_burn_in_time:g}, "
                f"tiempo de producción={time[-1]:g}."
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


        required_coordinate_names = [
            f"{axis}{monomer}"
            for monomer in range(N)
            for axis in ("x", "y", "z")
        ]


        missing_coordinate_names = [
            name
            for name in required_coordinate_names
            if name not in column_names
        ]


        if missing_coordinate_names:

            raise RuntimeError(
                "Faltan columnas de coordenadas "
                f"para N={N}: "
                f"{', '.join(missing_coordinate_names)}"
            )


        coordinate_columns = [
            column_names.index(
                name
            )
            for name in required_coordinate_names
        ]


        coordinates = (
            data[
                :,
                coordinate_columns
            ]
            .reshape(
                len(
                    data
                ),
                N,
                3
            )
        )


        center_of_mass = np.mean(
            coordinates,
            axis=1
        )

        center_of_mass_displacement = (
            center_of_mass
            - center_of_mass[0]
        )

        com_lag_indices, com_msd = (
            compute_time_averaged_msd(
                center_of_mass
            )
        )

        com_lag_times = np.array([
            np.mean(
                time[lag:]
                - time[:-lag]
            )
            for lag in com_lag_indices
        ])

        diffusion_fit_mask = (
            (com_lag_times >= 0.2 * com_lag_times[-1])
            & (com_lag_times <= 0.8 * com_lag_times[-1])
        )

        if np.count_nonzero(diffusion_fit_mask) < 2:

            raise RuntimeError(
                "No hay suficientes retardos para ajustar "
                f"la difusion del centro de masas en N={N}."
            )

        diffusion_slope, diffusion_intercept = np.polyfit(
            com_lag_times[diffusion_fit_mask],
            com_msd[diffusion_fit_mask],
            1
        )

        center_of_mass_diffusion_coefficient = float(
            diffusion_slope / 6.0
        )

        diffusion_fit_start_time = float(
            com_lag_times[diffusion_fit_mask][0]
        )

        diffusion_fit_end_time = float(
            com_lag_times[diffusion_fit_mask][-1]
        )


        centered_coordinates = (
            coordinates
            - center_of_mass[:, np.newaxis, :]
        )


        rg2_trajectory = np.mean(
            np.sum(
                centered_coordinates ** 2,
                axis=2
            ),
            axis=1
        )


        ree2_trajectory = np.sum(
            (
                coordinates[:, -1, :]
                - coordinates[:, 0, :]
            ) ** 2,
            axis=1
        )


        mean_rg2 = float(
            np.mean(
                rg2_trajectory[
                    equilibrium_mask
                ]
            )
        )

        rg_trajectory = np.sqrt(
            rg2_trajectory
        )


        mean_rg = float(
            np.mean(
                rg_trajectory[
                    equilibrium_mask
                ]
            )
        )


        mean_ree2 = float(
            np.mean(
                ree2_trajectory[
                    equilibrium_mask
                ]
            )
        )


        rg2_theory = (
            (N ** 2 - 1)
            / (6.0 * N)
            * bond_length_sq_theory
        )


        ree2_theory = (
            (N - 1)
            * bond_length_sq_theory
        )

        case_figures_folder = (
            figures_folder
            / f"N_{N}"
        )

        case_figures_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        plt.figure(
            figsize=(9, 6)
        )


        for axis_index, axis_label in enumerate(("x", "y", "z")):

            plt.plot(
                time,
                center_of_mass_displacement[:, axis_index],
                label=(
                    f"$R_{{CM,{axis_label}}}(t)"
                    f"-R_{{CM,{axis_label}}}(0)$"
                )
            )


        plt.xlabel(
            "Tiempo de producción $t$"
        )

        plt.ylabel(
            "Desplazamiento del centro de masas"
        )

        plt.title(
            f"Centro de masas del polímero ($N={N}$)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            case_figures_folder
            / f"polymer_center_of_mass_N_{N}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(
            time,
            rg_trajectory,
            label=(
                f"$R_g(t)$ "
                f"($\\langle R_g\\rangle={mean_rg:.4f}$)"
            )
        )

        plt.axhline(
            mean_rg,
            linestyle="--",
            label=f"$\\langle R_g\\rangle={mean_rg:.4f}$"
        )

        plt.axvspan(
            time[0],
            production_burn_in_time,
            color="gray",
            alpha=0.15,
            label="Tramo descartado"
        )

        plt.axvline(
            production_burn_in_time,
            color="gray",
            linestyle=":",
            label=(
                "Inicio del promedio: "
                f"$t={production_burn_in_time:g}$"
            )
        )

        plt.xlabel(
            "Tiempo de producción $t$"
        )

        plt.ylabel(
            "Radio de giro $R_g$"
        )

        plt.title(
            f"Radio de giro del polímero ($N={N}$)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            case_figures_folder
            / f"polymer_radius_of_gyration_N_{N}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        fig = plt.figure(
            figsize=(9, 7)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        trajectory_plot = ax.scatter(
            center_of_mass_displacement[:, 0],
            center_of_mass_displacement[:, 1],
            center_of_mass_displacement[:, 2],
            c=time,
            cmap="viridis",
            s=8
        )

        ax.plot(
            center_of_mass_displacement[:, 0],
            center_of_mass_displacement[:, 1],
            center_of_mass_displacement[:, 2],
            alpha=0.45,
            linewidth=0.8
        )

        ax.scatter(
            0.0,
            0.0,
            0.0,
            color="green",
            marker="o",
            s=45,
            label="Inicio"
        )

        ax.scatter(
            center_of_mass_displacement[-1, 0],
            center_of_mass_displacement[-1, 1],
            center_of_mass_displacement[-1, 2],
            color="red",
            marker="x",
            s=55,
            label="Final"
        )

        colorbar = fig.colorbar(
            trajectory_plot,
            ax=ax,
            pad=0.1
        )

        colorbar.set_label(
            "Tiempo de producción $t$"
        )

        ax.set_xlabel(
            "$R_{CM,x}(t)-R_{CM,x}(0)$"
        )

        ax.set_ylabel(
            "$R_{CM,y}(t)-R_{CM,y}(0)$"
        )

        ax.set_zlabel(
            "$R_{CM,z}(t)-R_{CM,z}(0)$"
        )

        ax.set_title(
            "Difusión 3D del centro de masas "
            f"($N={N}$)"
        )

        ax.legend()

        path_ranges = np.ptp(
            center_of_mass_displacement,
            axis=0
        )

        path_half_range = max(
            float(np.max(path_ranges)) / 2.0,
            1.0e-12
        )

        path_midpoint = (
            np.max(
                center_of_mass_displacement,
                axis=0
            )
            + np.min(
                center_of_mass_displacement,
                axis=0
            )
        ) / 2.0

        ax.set_xlim(
            path_midpoint[0] - path_half_range,
            path_midpoint[0] + path_half_range
        )

        ax.set_ylim(
            path_midpoint[1] - path_half_range,
            path_midpoint[1] + path_half_range
        )

        ax.set_zlim(
            path_midpoint[2] - path_half_range,
            path_midpoint[2] + path_half_range
        )

        plt.tight_layout()

        plt.savefig(
            case_figures_folder
            / f"polymer_com_diffusion_3d_N_{N}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(
            com_lag_times,
            com_msd,
            marker="o",
            markersize=3,
            label="MSD temporal del centro de masas"
        )

        plt.plot(
            com_lag_times[diffusion_fit_mask],
            (
                diffusion_slope
                * com_lag_times[diffusion_fit_mask]
                + diffusion_intercept
            ),
            linestyle="--",
            label=(
                f"Ajuste lineal: $D_{{CM}}="
                f"{center_of_mass_diffusion_coefficient:.4g}$"
            )
        )

        plt.xlabel(
            "Retardo temporal $\\tau$"
        )

        plt.ylabel(
            r"$\langle|\vec R_{CM}(t+\tau)-"
            r"\vec R_{CM}(t)|^2\rangle_t$"
        )

        plt.title(
            "Difusión del centro de masas en 3D "
            f"($N={N}$, pendiente $=6D_{{CM}}$)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            case_figures_folder
            / f"polymer_com_msd_N_{N}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        final_coordinates_centered = (
            coordinates[-1]
            - center_of_mass[-1]
        )

        fig = plt.figure(
            figsize=(8, 7)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.plot(
            final_coordinates_centered[:, 0],
            final_coordinates_centered[:, 1],
            final_coordinates_centered[:, 2],
            marker="o",
            markersize=5,
            label=f"Cadena ($N={N}$)"
        )

        ax.scatter(
            0.0,
            0.0,
            0.0,
            color="red",
            marker="x",
            s=65,
            label=r"$\vec R_{CM}=(0,0,0)$"
        )

        ax.set_xlabel(
            "$x-R_{CM,x}$"
        )

        ax.set_ylabel(
            "$y-R_{CM,y}$"
        )

        ax.set_zlabel(
            "$z-R_{CM,z}$"
        )

        ax.set_title(
            f"Conformación final centrada ($N={N}$)"
        )

        ax.legend()

        coil_ranges = np.ptp(
            final_coordinates_centered,
            axis=0
        )

        coil_half_range = max(
            float(np.max(coil_ranges)) / 2.0,
            1.0e-12
        )

        ax.set_xlim(
            -coil_half_range,
            coil_half_range
        )

        ax.set_ylim(
            -coil_half_range,
            coil_half_range
        )

        ax.set_zlim(
            -coil_half_range,
            coil_half_range
        )

        plt.tight_layout()

        plt.savefig(
            case_figures_folder
            / f"polymer_snapshot_3d_centered_N_{N}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        # ==========================
        # Energías
        # ===================================

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
            * (N - int(first_end == "fixed"))
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


        # ===================================
        # Longitudes de enlace
        # ===================================

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


        bond_length_per_frame = np.mean(
            bond_lengths,
            axis=1
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


        # ===================================
        # Incertidumbres por bloques
        # ===================================

        (
            kinetic_sem,
            kinetic_blocks,
            kinetic_block_size
        ) = block_standard_error(
            kinetic / N,
            number_of_blocks
        )


        (
            elastic_sem,
            elastic_blocks,
            elastic_block_size
        ) = block_standard_error(
            elastic / N,
            number_of_blocks
        )


        (
            bond_length_sem,
            bond_blocks,
            bond_block_size
        ) = block_standard_error(
            bond_length_per_frame,
            number_of_blocks
        )


        # ===================================
        # Convergencia acumulada
        # ===================================

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

            "kinetic_per_particle_sem":
                kinetic_sem,

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

            "elastic_per_particle_sem":
                elastic_sem,

            "mean_bond_length":
                mean_bond_length,

            "bond_length_theory":
                bond_length_theory,

            "bond_length_std":
                bond_length_std,

            "bond_length_sem":
                bond_length_sem,

            "number_of_blocks":
                bond_blocks,

            "block_size":
                bond_block_size,

            "mean_rg2":
                mean_rg2,

            "center_of_mass_diffusion_coefficient":
                center_of_mass_diffusion_coefficient,

            "diffusion_fit_start_time":
                diffusion_fit_start_time,

            "diffusion_fit_end_time":
                diffusion_fit_end_time,

            "equilibrium_sample_count":
                equilibrium_sample_count,

            "mean_rg":
                mean_rg,

            "rg2_theory":
                rg2_theory,

            "mean_ree2":
                mean_ree2,

            "ree2_theory":
                ree2_theory,

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

            "production_burn_in_time":
                production_burn_in_time,

            "time":
                time,

            "cumulative_kinetic":
                cumulative_kinetic,

            "cumulative_elastic":
                cumulative_elastic
        })


        # ===================================
        # Guardado de datos brutos
        # ===================================

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
            "Energia cinetica por particula:"
        )

        print(
            f"  exp = "
            f"{kinetic_per_particle:.6f} "
            f"+/- {kinetic_sem:.6f}"
        )

        print(
            f"  teo = "
            f"{kinetic_per_particle_theory:.6f}"
        )


        print(
            "Energia elastica por particula:"
        )

        print(
            f"  exp = "
            f"{elastic_per_particle:.6f} "
            f"+/- {elastic_sem:.6f}"
        )

        print(
            f"  teo = "
            f"{elastic_per_particle_theory:.6f}"
        )


        print(
            "Longitud media de enlace:"
        )

        print(
            f"  exp = "
            f"{mean_bond_length:.6f} "
            f"+/- {bond_length_sem:.6f}"
        )

        print(
            f"  teo = "
            f"{bond_length_theory:.6f}"
        )


finally:

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

        "mean_rg2":
            result[
                "mean_rg2"
            ],

        "center_of_mass_diffusion_coefficient":
            result[
                "center_of_mass_diffusion_coefficient"
            ],

        "diffusion_fit_start_time":
            result[
                "diffusion_fit_start_time"
            ],

        "diffusion_fit_end_time":
            result[
                "diffusion_fit_end_time"
            ],

        "equilibrium_sample_count":
            result[
                "equilibrium_sample_count"
            ],

        "mean_rg":
            result[
                "mean_rg"
            ],

        "rg2_theory":
            result[
                "rg2_theory"
            ],

        "mean_ree2":
            result[
                "mean_ree2"
            ],

        "ree2_theory":
            result[
                "ree2_theory"
            ],

        "bond_length_sem":
            result[
                "bond_length_sem"
            ],

        "number_of_blocks":
            result[
                "number_of_blocks"
            ],

        "block_size":
            result[
                "block_size"
            ],

        "thermalization_time":
            result[
                "thermalization_time"
            ],

        "saved_dt":
            result[
                "saved_dt"
            ],

        "production_burn_in_time":
            result[
                "production_burn_in_time"
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
        ),
        lineterminator="\n"
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ===================================
# Arrays para gráficas
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


kinetic_per_particle_sem = np.array([
    result[
        "kinetic_per_particle_sem"
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


elastic_per_particle_sem = np.array([
    result[
        "elastic_per_particle_sem"
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


bond_length_sem = np.array([
    result[
        "bond_length_sem"
    ]
    for result in results
])


# ===================================
# 1. Convergencia cinética / N
# ===================================

fig = plt.figure(
    figsize=(9, 6)
)


for result in results:

    kinetic_line, = plt.plot(
        result[
            "time"
        ],
        result[
            "cumulative_kinetic"
        ],
        linewidth=1.2,
        label=(
            f"$N={result['N']}$: "
            f"exp={result['kinetic_per_particle']:.3f}, "
            f"teo={result['kinetic_per_particle_theory']:.3f}"
        )
    )

    plt.axhline(
        result[
            "kinetic_per_particle_theory"
        ],
        color=kinetic_line.get_color(),
        linestyle="--",
        linewidth=0.9,
        alpha=0.7
    )


plt.xlabel(
    "Tiempo de producción $t$"
)


plt.ylabel(
    r"$\langle E_{cin}\rangle_{acum}/N$"
)


plt.title(
    "Convergencia de la energía cinética "
    "por partícula\n"
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
    / "polymer_kinetic_per_particle_convergence.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ===================================
# 2. Convergencia elástica / N
# ===================================

fig = plt.figure(
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
        linewidth=1.2,
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
    "por partícula\n"
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
    / "polymer_elastic_per_particle_convergence.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ===================================
# 3. Energías normalizadas vs N
# ===================================

fig = plt.figure(
    figsize=(9, 5.5)
)


# Puntos experimentales:
# NO se unen mediante líneas.

plt.errorbar(
    N_plot,
    kinetic_per_particle,
    yerr=kinetic_per_particle_sem,
    fmt="o",
    linestyle="none",
    markersize=7,
    capsize=4,
    label=(
        r"Simulación "
        r"$\langle E_{cin}\rangle/N$"
    )
)


plt.errorbar(
    N_plot,
    elastic_per_particle,
    yerr=elastic_per_particle_sem,
    fmt="s",
    linestyle="none",
    markersize=7,
    capsize=4,
    label=(
        r"Simulación "
        r"$\langle E_{el}\rangle/N$"
    )
)


# La teoría sí se representa
# mediante líneas.

plt.plot(
    N_plot,
    kinetic_per_particle_theory,
    linestyle="--",
    linewidth=1.8,
    label=(
        r"Teoría "
        r"$\langle E_{cin}\rangle/N"
        r"=\frac{3}{2}k_BT\frac{N-n_f}{N},\ "
        rf"n_f={int(first_end == 'fixed')}$"
    )
)


plt.plot(
    N_plot,
    elastic_per_particle_theory,
    linestyle="--",
    linewidth=1.8,
    label=(
        r"Teoría "
        r"$\langle E_{el}\rangle/N$"
    )
)


plt.xlabel(
    "Número de partículas $N$"
)


plt.ylabel(
    "Energía media por partícula"
)


plt.title(
    "Energías normalizadas frente "
    "al tamaño del polímero\n"
    f"$k_e={ke:g}$, "
    f"$l_0={l0:g}$, "
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
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ===================================
# 4. Longitud de enlace vs N
# ===================================

fig = plt.figure(
    figsize=(9, 5.5)
)


# Puntos experimentales:
# NO se unen mediante líneas.

plt.errorbar(
    N_plot,
    bond_length_mean,
    yerr=bond_length_sem,
    fmt="o",
    linestyle="none",
    markersize=7,
    capsize=4,
    label=(
        r"Simulación "
        r"$\langle l\rangle$"
    )
)


plt.axhline(
    bond_length_theory,
    linestyle="--",
    linewidth=1.8,
    label=(
        r"Teoría "
        r"$\langle l\rangle$"
        f" = {bond_length_theory:.4f}"
    )
)


plt.axhline(
    l0,
    linestyle=":",
    linewidth=1.8,
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
    "Longitud media de enlace frente "
    "al tamaño del polímero\n"
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
    / "polymer_bond_length_vs_N.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


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
    f"T produccion = "
    f"{total_time}"
)

print(
    "T minimo termalizacion = "
    f"{min_thermalization_time}"
)

print(
    "Tiempo de produccion descartado "
    "para promedios estructurales = "
    f"{production_burn_in_time}"
)

print()

print(
    "Teoria:"
)

print(
    f"  <Ecin>/N = "
    f"3kBT(N-{int(first_end == 'fixed')})/(2N), "
    f"firstEnd={first_end}"
)

print(
    f"  <U enlace> = "
    f"{elastic_per_bond_theory:.6f}"
)

print(
    f"  <l> = "
    f"{bond_length_theory:.6f}"
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
        f"{result['kinetic_per_particle']:.6f} "
        f"+/- "
        f"{result['kinetic_per_particle_sem']:.6f}"
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
        f"{result['elastic_per_particle']:.6f} "
        f"+/- "
        f"{result['elastic_per_particle_sem']:.6f}"
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
        f"{result['mean_bond_length']:.6f} "
        f"+/- "
        f"{result['bond_length_sem']:.6f}"
    )

    print(
        f"    teo = "
        f"{bond_length_theory:.6f}"
    )

    print(
        "  <Rg^2> = "
        f"{result['mean_rg2']:.6f} "
        "(FJC: "
        f"{result['rg2_theory']:.6f})"
    )

    print(
        "    calculado desde t = "
        f"{production_burn_in_time:g} "
        f"({result['equilibrium_sample_count']} muestras)"
    )

    print(
        "  <Rg> = "
        f"{result['mean_rg']:.6f}"
    )

    print(
        "  <Ree^2> = "
        f"{result['mean_ree2']:.6f} "
        "(FJC: "
        f"{result['ree2_theory']:.6f})"
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

rg2_mean = np.array([
    result["mean_rg2"]
    for result in results
])

rg2_theory_arr = np.array([
    result["rg2_theory"]
    for result in results
])

rg_rms = np.sqrt(rg2_mean)

ree2_mean = np.array([
    result["mean_ree2"]
    for result in results
])

ree2_theory_arr = np.array([
    result["ree2_theory"]
    for result in results
])

# Ajuste de ley de escala: R_G ~ N^nu  (o Rg^2 ~ N^(2*nu))
# Usamos (N - 1) o N en log-log
log_N = np.log(N_plot)
log_rg = np.log(rg_rms)

nu_fit, log_a_fit = np.polyfit(
    log_N,
    log_rg,
    1
)


scaling_summary_file = (
    tables_folder
    / "polymer_scaling_summary.txt"
)


nu_difference = abs(
    nu_fit - 0.5
)


if nu_difference < 0.05:

    scaling_conclusion = (
        "El exponente ajustado es compatible con "
        "nu = 0.5, esperado para una cadena ideal "
        "tipo random walk."
    )

else:

    scaling_conclusion = (
        "El exponente ajustado se aleja de nu = 0.5. "
        "No se confirma la ley esperada con este barrido; "
        "revisa el equilibrio, el muestreo y las "
        "fluctuaciones estadisticas."
    )


scaling_summary_file.write_text(
    "COMPROBACION DE LA LEY DE ESCALA DEL POLIMERO\n"
    "\n"
    f"N evaluados = {', '.join(str(value) for value in N_plot)}\n"
    f"Tiempo inicial descartado = "
    f"{production_burn_in_time:.6f}\n"
    "Promedio calculado solo en el regimen posterior al descarte.\n"
    f"<Rg^2> medio por N = "
    f"{', '.join(f'{value:.8f}' for value in rg2_mean)}\n"
    f"Exponente ajustado nu = {nu_fit:.6f}\n"
    "Exponente esperado para cadena ideal = 0.5\n"
    f"Diferencia respecto a 0.5 = {nu_difference:.6f}\n"
    "Difusion del centro de masas (D_CM = pendiente MSD/6):\n"
    + "".join(
        f"  N={result['N']}: "
        f"D_CM={result['center_of_mass_diffusion_coefficient']:.8g}, "
        f"ajuste en tau=[{result['diffusion_fit_start_time']:.6g}, "
        f"{result['diffusion_fit_end_time']:.6g}]\n"
        for result in results
    )
    + f"Conclusión: {scaling_conclusion}\n"
)


print()

print(
    "Comprobacion de la ley de escala:"
)

print(
    "  Rg ~ N^nu"
)

print(
    f"  nu ajustado = {nu_fit:.4f}"
)

print(
    f"  diferencia respecto a 0.5 = {nu_difference:.4f}"
)

print(
    "  D_CM estimado por N:"
)

for result in results:

    print(
        f"    N={result['N']}: "
        f"{result['center_of_mass_diffusion_coefficient']:.6g} "
        f"(ajuste tau="
        f"{result['diffusion_fit_start_time']:.4g}–"
        f"{result['diffusion_fit_end_time']:.4g})"
    )

print(
    "  nu esperado (cadena ideal) = 0.5"
)

print(
    f"  {scaling_conclusion}"
)

print(
    "  Resumen:"
)

print(
    scaling_summary_file
)


# ===================================
# 5. <Rg^2> frente a N (escala lineal)
# ===================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    N_plot,
    rg2_mean,
    marker="o",
    label=r"Simulación $\langle R_G^2\rangle$"
)

plt.plot(
    N_plot,
    rg2_theory_arr,
    linestyle="--",
    label=r"Cadena ideal $\frac{N^2-1}{6N}\langle l^2\rangle$"
)

plt.xlabel(
    "Número de monómeros $N$"
)

plt.ylabel(
    r"$\langle R_G^2\rangle$"
)

plt.title(
    r"Radio de giro cuadrático medio frente a $N$"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    figures_folder
    / "polymer_rg2_vs_N.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ===================================
# 6. Ley de escala: Rg^2 y Ree^2 vs N
# ===================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    N_plot,
    rg2_mean,
    marker="o",
    label=r"Simulación $\langle R_G^2\rangle$"
)

plt.plot(
    N_plot,
    rg2_theory_arr,
    linestyle="--",
    label=r"Teoría FJC $\frac{N^2-1}{6N}\langle l^2\rangle$"
)

plt.plot(
    N_plot,
    ree2_mean,
    marker="s",
    label=r"Simulación $\langle R_{ee}^2\rangle$"
)

plt.plot(
    N_plot,
    ree2_theory_arr,
    linestyle=":",
    label=r"Teoría FJC $(N-1)\langle l^2\rangle$"
)

plt.xlabel(
    "Número de monómeros $N$"
)

plt.ylabel(
    r"Distancia cuadrática media"
)

plt.title(
    "Crecimiento lineal de $\\langle R_G^2\\rangle$ y "
    "$\\langle R_{ee}^2\\rangle$ con $N$"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    figures_folder
    / "polymer_rg2_ree2_vs_N.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ===================================
# 7. Ley de escala Log-Log (Exponente nu)
# ===================================

plt.figure(
    figsize=(8, 5)
)

plt.loglog(
    N_plot,
    rg_rms,
    marker="o",
    linestyle="None",
    label=r"Simulación $\sqrt{\langle R_G^2\rangle}$"
)

N_fine = np.linspace(
    min(N_plot),
    max(N_plot),
    200
)

plt.loglog(
    N_fine,
    np.exp(log_a_fit) * (N_fine ** nu_fit),
    linestyle="-",
    label=(
        f"Ajuste ley de escala: "
        f"$R_G \\propto N^{{\\nu}}$ "
        f"($\\nu = {nu_fit:.3f}$)"
    )
)

plt.loglog(
    N_fine,
    np.sqrt((N_fine ** 2 - 1) / (6.0 * N_fine) * bond_length_sq_theory),
    linestyle="--",
    label=r"Teoría cadena ideal ($\nu \to 0.5$)"
)

plt.xlabel(
    "Número de monómeros $N$ (escala log)"
)

plt.ylabel(
    r"Radio de giro $\sqrt{\langle R_G^2\rangle}$ (escala log)"
)

plt.title(
    "Ley de escala del radio de giro del polímero"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    figures_folder
    / "polymer_rg_scaling_loglog.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()