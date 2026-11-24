from pathlib import Path

import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from polymer_metrics import compute_time_averaged_msd


project_root = Path(
    __file__
).resolve().parents[1]


parameters_file = (
    project_root
    / "data"
    / "raw"
    / "polymer"
    / "parameters.txt"
)


data_file = (
    project_root
    / "data"
    / "raw"
    / "polymer"
    / "polymer.csv"
)


figures_folder = (
    project_root
    / "results"
    / "polymer"
    / "single_run"
    / "figures"
)


figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ===================================
# Funciones auxiliares
# ===================================

def integrate(y, x):

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
# Parámetros
# ===================================

parameters = {}


with parameters_file.open() as file:

    for line in file:

        line = line.strip()

        if not line:

            continue

        if "=" not in line:

            continue

        name, value = (
            line.split(
                "=",
                maxsplit=1
            )
        )

        parameters[
            name
        ] = (
            value
            if name == "firstEnd"
            else float(value)
        )


N = int(
    parameters[
        "N"
    ]
)

ke = parameters[
    "ke"
]


l0 = parameters[
    "l0"
]

force = parameters[
    "force"
]


mass = parameters[
    "mass"
]


gamma = parameters[
    "gamma"
]


kBT = parameters[
    "kBT"
]


dt = parameters[
    "dt"
]


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


if len(data) < 2:

    raise RuntimeError(
        "No hay suficientes datos."
    )


if not np.all(
    np.isfinite(
        data
    )
):

    raise RuntimeError(
        "Hay valores no finitos "
        "en la simulacion."
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
        "Faltan columnas de coordenadas: "
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
        len(data),
        N,
        3
    )
)


# ===================================
# Valores experimentales
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


mean_total = float(
    np.mean(
        kinetic
        + elastic
    )
)


# ===================================
# Valores teóricos
# ===================================

# Energía cinética:
#
# <Ecin> = 3 N_libres kBT / 2

free_monomers = N - int(parameters.get("firstEnd", "fixed") == "fixed")
kinetic_theory = (
    1.5
    * free_monomers
    * kBT
)


# Para un enlace bajo una fuerza en +x:
#
# U(l, theta) = U_resorte(l) - F l cos(theta)
#
# Al integrar las orientaciones en 3D aparece sinh(q)/q,
# donde q = F l / kBT.

sigma = np.sqrt(
    kBT
    / ke
)


l_grid = np.linspace(
    0.0,
    l0
    + abs(force)
    / ke
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


force_argument = (
    abs(force)
    * l_grid
    / kBT
)


log_angular_weight = np.empty_like(
    force_argument
)

small_argument = (
    force_argument
    < 0.1
)

small_argument_squared = (
    force_argument[small_argument] ** 2
)

log_angular_weight[small_argument] = (
    small_argument_squared / 6.0
    - small_argument_squared ** 2 / 180.0
    + small_argument_squared ** 3 / 2835.0
)

large_argument = ~small_argument
large_force_argument = force_argument[large_argument]

log_angular_weight[large_argument] = (
    large_force_argument
    - np.log(
        2.0
        * large_force_argument
    )
    + np.log1p(
        -np.exp(
            -2.0
            * large_force_argument
        )
    )
)


log_radial_weight = np.full_like(
    l_grid,
    -np.inf
)

positive_lengths = (
    l_grid
    > 0.0
)

log_radial_weight[positive_lengths] = (
    2.0
    * np.log(
        l_grid[positive_lengths]
    )
)


log_bond_weight = (
    log_radial_weight
    - bond_potential / kBT
    + log_angular_weight
)


bond_weight = np.exp(
    log_bond_weight
    - np.max(
        log_bond_weight
    )
)

bond_weight_without_force = (
    l_grid ** 2
    * np.exp(
        -bond_potential
        / kBT
    )
)

normalization = integrate(
    bond_weight,
    l_grid
)
normalization_without_force = integrate(
    bond_weight_without_force,
    l_grid
)


bond_probability_theory = (
    bond_weight
    / normalization
)
bond_probability_theory_without_force = (
    bond_weight_without_force
    / normalization_without_force
)


bond_length_theory = float(
    integrate(
        l_grid
        * bond_probability_theory,
        l_grid
    )
)
bond_length_theory_without_force = float(
    integrate(
        l_grid
        * bond_probability_theory_without_force,
        l_grid
    )
)


elastic_per_bond_theory = float(
    integrate(
        bond_potential
        * bond_probability_theory,
        l_grid
    )
)
elastic_per_bond_theory_without_force = float(
    integrate(
        bond_potential
        * bond_probability_theory_without_force,
        l_grid
    )
)


elastic_theory = (
    (N - 1)
    * elastic_per_bond_theory
)

elastic_theory_without_force = (
    (N - 1)
     * elastic_per_bond_theory_without_force
     
)


# ===================================
# Errores relativos
# ===================================

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
elastic_error_without_force = (
    abs(
        mean_elastic
        - elastic_theory_without_force
    )
    / elastic_theory_without_force
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


mean_bond_length = float(
    np.mean(
        bond_lengths
    )
)


std_bond_length = float(
    np.std(
        bond_lengths
    )
)


bond_length_error = (
    abs(
        mean_bond_length
        - bond_length_theory
    )
    / bond_length_theory
    * 100.0
)
bond_length_error_without_force = (
    abs(
        mean_bond_length
        - bond_length_theory_without_force
    )
    / bond_length_theory_without_force
    * 100.0
)


# ===================================
# Límites compartidos con la animación
# ===================================

max_animation_frames = 200

animation_fps = 20


number_animation_frames = min(
    max_animation_frames,
    len(
        coordinates
    )
)


animation_indices = np.unique(
    np.linspace(
        len(
            coordinates
        ) - max_animation_frames,
        len(
            coordinates
        ) - 1,
        number_animation_frames,
        dtype=int
    )
)


animation_coordinates = (
    coordinates[
        animation_indices
    ]
    .copy()
)


coordinate_min = np.min(
    animation_coordinates,
    axis=(0, 1)
)

coordinate_max = np.max(
    animation_coordinates,
    axis=(0, 1)
)

coordinate_span = np.maximum(
    coordinate_max - coordinate_min,
    1.0
)

axis_padding = 0.05 * coordinate_span
axis_min = coordinate_min - axis_padding
axis_max = coordinate_max + axis_padding


# ===================================
# 1. Configuración final 3D
# ===================================

final_coordinates = (
    coordinates[
        -1
    ]
    .copy()
)


center_of_mass = np.mean(
    final_coordinates,
    axis=0
)


final_coordinates -= (
    center_of_mass
)


fig = plt.figure(
    figsize=(7, 6)
)


ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.plot(
    final_coordinates[
        :,
        0
    ],
    final_coordinates[
        :,
        1
    ],
    final_coordinates[
        :,
        2
    ],
    marker=".",
    linewidth=1.2,
    label=f"$N={N}$"
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
    "Configuración final del polímero\n"
    f"$N={N}$, "
    f"$k_e={ke:g}$, "
    f"$l_0={l0:g}$, "
    f"$k_BT={kBT:g}$"
    f"$Fuerza={force:g}$"
)


snapshot_axis_half_range = (
    axis_max
    - axis_min
) / 2.0


ax.set_xlim(
    -snapshot_axis_half_range[0],
    snapshot_axis_half_range[0]
)


ax.set_ylim(
    -snapshot_axis_half_range[1],
    snapshot_axis_half_range[1]
)


ax.set_zlim(
    -snapshot_axis_half_range[2],
    snapshot_axis_half_range[2]
)


ax.set_box_aspect(
    (
        1,
        1,
        1
    )
)



plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_snapshot_3d.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ===================================
# 1b. Animación 3D del polímero
# ===================================

# Para poner el centro de masas en el origen:

# # Centramos cada configuración
# # en su propio centro de masas.
# #
# # Así eliminamos la traslación global
# # browniana y observamos principalmente
# # los cambios de forma del polímero.

# animation_centers = np.mean(
#     animation_coordinates,
#     axis=1,
#     keepdims=True
# )


# animation_coordinates -= (
#     animation_centers
# )

fig_animation = plt.figure(
    figsize=(7, 6)
)


ax_animation = (
    fig_animation
    .add_subplot(
        111,
        projection="3d"
    )
)


polymer_line, = ax_animation.plot(
    [],
    [],
    [],
    marker=".",
    linewidth=1.5
)


time_text = ax_animation.text2D(
    0.03,
    0.95,
    "",
    transform=ax_animation.transAxes
)


ax_animation.set_xlim(
    axis_min[0],
    axis_max[0]
)


ax_animation.set_ylim(
    axis_min[1],
    axis_max[1]
)


ax_animation.set_zlim(
    axis_min[2],
    axis_max[2]
)


ax_animation.set_box_aspect(
    (1,1,1)
)

ax_animation.set_xlabel(
    "$x$"
)


ax_animation.set_ylabel(
    "$y$"
)


ax_animation.set_zlabel(
    "$z$"
)


ax_animation.set_title(
    "Dinámica del polímero\n"
    f"$N={N}$, "
    f"$k_e={ke:g}$, "
    f"$k_BT={kBT:g}$, "
    f"F={force:g}"
)


def update_animation(frame):

    current_coordinates = (
        animation_coordinates[
            frame
        ]
    )


    polymer_line.set_data(
        current_coordinates[
            :,
            0
        ],
        current_coordinates[
            :,
            1
        ]
    )


    polymer_line.set_3d_properties(
        current_coordinates[
            :,
            2
        ]
    )


    original_index = (
        animation_indices[
            frame
        ]
    )


    time_text.set_text(
        f"$t={time[original_index]:.1f}$"
    )


    return (
        polymer_line,
        time_text
    )


animation_frames = np.concatenate(
    (
        np.arange(
            len(
                animation_indices
            )
        ),
        np.full(
            animation_fps,
            len(
                animation_indices
            )
            - 1
        )
    )
)


polymer_animation = FuncAnimation(
    fig_animation,
    update_animation,
    frames=animation_frames,
    interval=(
        1000
        / animation_fps
    ),
    blit=False
)


polymer_animation.save(
    figures_folder
    / "polymer_motion_3d.gif",
    writer=PillowWriter(
        fps=animation_fps
    ),
    dpi=120
)


plt.close(
    fig_animation
)


# ===================================
# 2. Energías instantáneas
# ===================================

fig = plt.figure(
    figsize=(9, 5.5)
)


plt.plot(
    time,
    kinetic,
    linewidth=0.7,
    alpha=0.35,
    label="$E_{cin}(t)$"
)


plt.plot(
    time,
    elastic,
    linewidth=0.7,
    alpha=0.35,
    label="$E_{el}(t)$"
)


plt.axhline(
    kinetic_theory,
    linestyle="--",
    linewidth=1.8,
    label=(
        r"$\langle E_{cin}\rangle_{teo}$"
        f" = {kinetic_theory:.3f}"
    )
)


plt.axhline(
    elastic_theory,
    linestyle=":",
    linewidth=1.8,
    label=(
        r"$\langle E_{el}\rangle_{teo}$"
        f" = {elastic_theory:.3f}"
    )
)


plt.xlabel(
    "Tiempo de producción $t$"
)


plt.ylabel(
    "Energía"
)


plt.title(
    "Energías instantáneas del polímero\n"
    f"$N={N}$, "
    f"$k_e={ke:g}$, "
    f"$k_BT={kBT:g}$"
    f"$F={force:g}$"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend(
    title=(
        r"$\langle E_{cin}\rangle_{exp}$"
        f" = {mean_kinetic:.3f}\n"
        r"$\langle E_{el}\rangle_{exp}$"
        f" = {mean_elastic:.3f}"
    )
)


plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_energies.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ===================================
# 3. Convergencia de energías
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
)


cumulative_elastic = (
    np.cumsum(
        elastic
    )
    / sample_count
)

cumulative_kinetic_normalized = (
    cumulative_kinetic
    / free_monomers
)

cumulative_elastic_normalized = (
    cumulative_elastic
    / (N - 1)
)


fig = plt.figure(
    figsize=(9, 5.5)
)


plt.plot(
    time,
    cumulative_kinetic_normalized,
    linewidth=1.4,
    label=(
        r"$\langle E_{cin}\rangle_{acum}$"
        f" → {mean_kinetic/free_monomers:.3f}"
    )
)


plt.plot(
    time,
    cumulative_elastic_normalized,
    linewidth=1.4,
    label=(
        r"$\langle E_{el}\rangle_{acum}$"
        f" → {mean_elastic/(N-1):.3f}"
    )
)


plt.axhline(
    kinetic_theory / free_monomers,
    linestyle="--",
    linewidth=1.8,
    label=(
        r"$\langle E_{cin}\rangle_{teo}$"
        f" = {kinetic_theory/free_monomers:.3f}"
    )
)


plt.axhline(
    elastic_theory / (N-1),
    linestyle=":",
    color="tab:orange",
    linewidth=1.8,
    label=(
        r"$\langle E_{el}\rangle_{teo}$"
        f" = {elastic_theory/(N-1):.3f}"
    )
)
if(force!=0):
    plt.axhline(
        elastic_theory_without_force / (N-1),
        linestyle="-.",
        linewidth=1.8,
        label=(
            r"$\langle E_{el}\rangle_{teo}$"
            f"$ sin fuerza = {elastic_theory_without_force/(N-1):.3f}$"
        )
    )


plt.xlabel(
    "Tiempo de producción $t$"
)


plt.ylabel(
    "Energía media acumulada"
)


plt.title(
    "Convergencia de las energías\n"
    f"$N={N}$, "
    f"$k_e={ke:g}$, "
    f"$k_BT={kBT:g}, $"
    f"$F={force:g}, $"
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend(
    title=f"$F={force:g}$"
)


plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_energy_convergence.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ===================================
# 4. Distribución de longitudes
# ===================================

fig = plt.figure(
    figsize=(9, 5.5)
)


plt.hist(
    bond_lengths.flatten(),
    bins=60,
    density=True,
    alpha=0.55,
    label="Simulación"
)


plt.plot(
    l_grid,
    bond_probability_theory,
    linewidth=2.0,
    label=(
        r"Teoría "
        r"$P(l)\propto "
        r"l^2e^{-U_{\rm resorte}(l)/k_BT}"
        r"\frac{\sinh(q)}{q}$"
    )
)


plt.axvline(
    l0,
    linestyle="--",
    linewidth=1.7,
    label=(
        f"$l_0={l0:g}$"
    )
)


plt.axvline(
    bond_length_theory,
    linestyle="-.",
    linewidth=1.7,
    label=(
        r"$\langle l\rangle_{teo}$"
        f" = {bond_length_theory:.4f}"
    )
)


plt.axvline(
    mean_bond_length,
    linestyle=":",
    linewidth=2.0,
    label=(
        r"$\langle l\rangle_{exp}$"
        f" = {mean_bond_length:.4f}"
    )
)
plt.legend()

plt.xlabel(
    "Longitud de enlace $l$"
)


plt.ylabel(
    "Densidad de probabilidad"
)


plt.title(
    "Distribución de longitudes de enlace\n"
    f"$N={N}$, "
    f"$k_e={ke:g}$, "
    f"$l_0={l0:g}$, "
    f"$k_BT={kBT:g}$, "
    f"$F={force:g}$"
)


plt.grid(
    True,
    alpha=0.3
)



plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_bond_lengths.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ==============================
# Centro de masas, Rg y Ree
# ==============================

center_of_mass_t = np.mean(
    coordinates,
    axis=1
)

displacements_cm = (
    coordinates
    - center_of_mass_t[:, np.newaxis, :]
)

rg2_t = np.mean(
    np.sum(
        displacements_cm ** 2,
        axis=2
    ),
    axis=1
)

rg_t = np.sqrt(
    rg2_t
)

ree_vectors = (
    coordinates[:, -1, :]
    - coordinates[:, 0, :]
)

ree_t = np.linalg.norm(
    ree_vectors,
    axis=1
)

ree2_t = ree_t ** 2

mean_rg2 = float(
    np.mean(
        rg2_t
    )
)

mean_rg = float(
    np.mean(
        rg_t
    )
)

mean_ree2 = float(
    np.mean(
        ree2_t
    )
)

mean_ree = float(
    np.mean(
        ree_t
    )
)

center_of_mass_displacement = (
    center_of_mass_t
    - center_of_mass_t[0]
)

com_lag_indices, com_msd = (
    compute_time_averaged_msd(
        center_of_mass_t
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
        "la difusion del centro de masas."
    )

diffusion_slope, diffusion_intercept = np.polyfit(
    com_lag_times[diffusion_fit_mask],
    com_msd[diffusion_fit_mask],
    1
)

center_of_mass_diffusion_coefficient = float(
    diffusion_slope / 6.0
)


bond_length_sq_theory = float(
    np.trapezoid(
        (l_grid ** 2)
        * bond_weight,
        l_grid
    )
    / normalization
)

ree2_theory = (
    (N - 1)
    * bond_length_sq_theory
)

rg2_theory = (
    (N ** 2 - 1)
    / (6.0 * N)
    * bond_length_sq_theory
)


# ==============================
# Resumen
# ===================================

summary_file = (
    project_root
    / "results"
    / "polymer"
    / "single_run"
    / "polymer_summary.txt"
)


summary_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


summary_text = (
    "POLYMER SIMULATION SUMMARY\n"
    "\n"
    f"N = {N}\n"
    f"ke = {ke}\n"
    f"l0 = {l0}\n"
    f"kBT = {kBT}\n"
    f"eta = {gamma}\n"
    f"h = {dt}\n"
    "\n"
    "Center of mass (final) = "
    f"({center_of_mass_t[-1, 0]:.6f}, "
    f"{center_of_mass_t[-1, 1]:.6f}, "
    f"{center_of_mass_t[-1, 2]:.6f})\n"
    f"<radius of gyration> = {mean_rg:.6f}\n"
    f"std radius of gyration = "
    f"{np.std(rg_t):.6f}\n"
    f"final radius of gyration = {rg_t[-1]:.6f}\n"
    f"<radius of gyration squared> = {mean_rg2:.6f}\n"
    f"<end-to-end distance> = {mean_ree:.6f}\n"
    f"<end-to-end distance squared> = {mean_ree2:.6f}\n"
    f"Center-of-mass diffusion coefficient = "
    f"{center_of_mass_diffusion_coefficient:.6f}\n"
    "\n"
    f"<Ekin> experimental = {mean_kinetic:.6f}\n"
    f"<Ekin> theoretical = {kinetic_theory:.6f}\n"
    f"relative error = {kinetic_error:.3f} %\n"
    "\n"
    f"<Eelastic> experimental = "
    f"{mean_elastic:.6f}\n"
    f"<Eelastic> theoretical = "
    f"{elastic_theory:.6f}\n"
    f"<Eelastic>/N experimental = "
    f"{mean_elastic/N:.6f}\n"
    f"<Eelastic>/N theoretical = "
    f"{elastic_theory/N:.6f}\n"
    f"relative error = "
    f"{elastic_error:.3f} %\n"
    "\n"
    f"<bond length> experimental = "
    f"{mean_bond_length:.6f}\n"
    f"<bond length> theoretical = "
    f"{bond_length_theory:.6f}\n"
    f"relative error = "
    f"{bond_length_error:.3f} %\n"
    f"std bond length = "
    f"{std_bond_length:.6f}\n"
)


summary_file.write_text(
    summary_text
)


# ===================================
# Información por terminal
# ===================================

print()

print(
    "==================================="
)

print(
    "POLYMER RESULTS"
)

print(
    "==================================="
)

print()

print(
    f"N = {N}"
)

print(
    f"ke = {ke}"
)

print(
    f"l0 = {l0}"
)

print(
    f"kBT = {kBT}"
)

print()

print(
    "Energia cinetica:"
)

print(
    f"  Experimental = "
    f"{mean_kinetic:.6f}"
)

print(
    f"  Teorica      = "
    f"{kinetic_theory:.6f}"
)

print(
    f"  Experimental / N = "
    f"{mean_kinetic/N:.6f}"
)

print(
    f"  Teorica / N      = "
    f"{kinetic_theory/N:.6f}"
)

print(
    f"  Error = "
    f"{kinetic_error:.3f} %"
)

print()

print(
    "Energia elastica:"
)

print(
    f"  Experimental = "
    f"{mean_elastic:.6f}"
)

print(
    f"  Teorica      = "
    f"{elastic_theory:.6f}"
)

print(
    f"  Experimental / N = "
    f"{mean_elastic/N:.6f}"
)

print(
    f"  Teorica / N      = "
    f"{elastic_theory/N:.6f}"
)

print(
    f"  Error = "
    f"{elastic_error:.3f} %"
)

print()

print(
    "Longitud de enlace:"
)

print(
    f"  Experimental = "
    f"{mean_bond_length:.6f}"
)

print(
    f"  Teorica      = "
    f"{bond_length_theory:.6f}"
)

print(
    f"  Error        = "
    f"{bond_length_error:.3f} %"
)

print(
    f"  std(l)       = "
    f"{std_bond_length:.6f}"
)

print()

print(
    "GIF:"
)

print(
    figures_folder
    / "polymer_motion_3d.gif"
)

print()

print(
    "Figuras guardadas en:"
)

print(
    figures_folder
)


# ==============================
# Centro de masas, Rg y Ree
# ==============================

# ==============================
# 5. Evolución de Rg^2 y Ree
# ==============================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 7),
    sharex=True
)

ax1.plot(
    time,
    rg2_t,
    alpha=0.7,
    label=(
        r"$R_G^2(t)$ "
        f"($\\langle R_G^2\\rangle_{{exp}}={mean_rg2:.4f}$)"
    )
)

ax1.axhline(
    rg2_theory,
    linestyle="--",
    color="black",
    label=(
        r"$\langle R_G^2\rangle_{teo}$"
        f" = {rg2_theory:.4f}"
    )
)

ax1.set_ylabel(r"$R_G^2$")
ax1.set_title("Evolución del radio de giro y distancia extremo a extremo")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(
    time,
    ree_t,
    color="tab:orange",
    alpha=0.7,
    label=(
        r"$R_{ee}(t)$ "
        f"($\\langle R_{{ee}}\\rangle_{{exp}}={mean_ree:.4f}$)"
    )
)

ax2.axhline(
    np.sqrt(ree2_theory),
    linestyle="--",
    color="black",
    label=(
        r"$\sqrt{\langle R_{ee}^2\rangle_{teo}}$"
        f" = {np.sqrt(ree2_theory):.4f}"
    )
)

ax2.set_xlabel("Tiempo $t$")
ax2.set_ylabel(r"$R_{ee}$")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()

plt.savefig(
    figures_folder
    / "polymer_rg_ree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(
    figsize=(8, 5)
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
    "Tiempo $t$"
)

plt.ylabel(
    "Desplazamiento del centro de masas"
)

plt.title(
    "Desplazamiento del centro de masas del polímero"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    figures_folder
    / "polymer_center_of_mass.png",
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
    marker=".",
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
    "Tiempo $t$"
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
    "Difusión 3D del centro de masas"
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
    figures_folder
    / "polymer_com_diffusion_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



plt.close()


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    time,
    rg_t,
    label=(
        f"$R_g$ "
        f"($\\langle R_g\\rangle={mean_rg:.4f}$)"
    )
)

plt.axhline(
    mean_rg,
    linestyle="--",
    label=f"$\\langle R_g\\rangle={mean_rg:.4f}$"
)

plt.xlabel(
    "Tiempo $t$"
)

plt.ylabel(
    "Radio de giro $R_g$"
)

plt.title(
    "Radio de giro del polímero"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    figures_folder
    / "polymer_radius_of_gyration.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()