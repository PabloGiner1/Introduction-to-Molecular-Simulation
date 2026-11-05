from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

        name, value = (
            line.strip()
            .split(
                "=",
                maxsplit=1
            )
        )

        parameters[
            name
        ] = float(
            value
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


# ===================================
# Datos
# ===================================

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
# Teoría
# ===================================

# Energía cinética:
#
# <Ecin> = 3 N kBT / 2

kinetic_theory = (
    1.5
    * N
    * kBT
)


# Distribución radial de un enlace:
#
# P(l) ~ l^2 exp[-U(l)/kBT]
#
# U(l) = ke (l-l0)^2 / 2

sigma = np.sqrt(
    kBT
    / ke
)


l_grid = np.linspace(
    max(
        0.0,
        l0 - 6.0 * sigma
    ),
    l0 + 6.0 * sigma,
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


normalization = integrate(
    bond_weight,
    l_grid
)


bond_probability_theory = (
    bond_weight
    / normalization
)


bond_length_theory = float(
    integrate(
        l_grid
        * bond_probability_theory,
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


elastic_theory = (
    (N - 1)
    * elastic_per_bond_theory
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
    marker="o",
    linewidth=1.2,
    label=(
        f"$N={N}$"
    )
)


ax.set_xlabel(
    "$x$"
)


ax.set_ylabel(
    "$y$"
)


ax.set_zlabel(
    "$z$"
)


ax.set_title(
    "Configuración final del polímero\n"
    f"$N={N}$, "
    f"$k_e={ke:g}$, "
    f"$l_0={l0:g}$, "
    f"$k_BT={kBT:g}$"
)


ranges = np.array([
    np.ptp(
        final_coordinates[
            :,
            0
        ]
    ),

    np.ptp(
        final_coordinates[
            :,
            1
        ]
    ),

    np.ptp(
        final_coordinates[
            :,
            2
        ]
    )
])


max_range = max(
    np.max(
        ranges
    ),
    1.0
)


midpoints = np.mean(
    np.array([
        np.min(
            final_coordinates,
            axis=0
        ),
        np.max(
            final_coordinates,
            axis=0
        )
    ]),
    axis=0
)


ax.set_xlim(
    midpoints[0]
    - max_range / 2,
    midpoints[0]
    + max_range / 2
)


ax.set_ylim(
    midpoints[1]
    - max_range / 2,
    midpoints[1]
    + max_range / 2
)


ax.set_zlim(
    midpoints[2]
    - max_range / 2,
    midpoints[2]
    + max_range / 2
)


ax.legend()


plt.tight_layout()


plt.savefig(
    figures_folder
    / "polymer_snapshot_3d.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 2. Energías instantáneas
# ===================================

plt.figure(
    figsize=(9, 5.5)
)


plt.plot(
    time,
    kinetic,
    linewidth=0.7,
    alpha=0.35,
    label=(
        "$E_{cin}(t)$"
    )
)


plt.plot(
    time,
    elastic,
    linewidth=0.7,
    alpha=0.35,
    label=(
        "$E_{el}(t)$"
    )
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


plt.close()


# ===================================
# 3. Convergencia de energías
# ===================================

sample_count = (
    np.arange(
        len(time)
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


plt.figure(
    figsize=(9, 5.5)
)


plt.plot(
    time,
    cumulative_kinetic,
    linewidth=1.4,
    label=(
        r"$\langle E_{cin}\rangle_{acum}$"
        f" → {mean_kinetic:.3f}"
    )
)


plt.plot(
    time,
    cumulative_elastic,
    linewidth=1.4,
    label=(
        r"$\langle E_{el}\rangle_{acum}$"
        f" → {mean_elastic:.3f}"
    )
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
    "Energía media acumulada"
)


plt.title(
    "Convergencia de las energías\n"
    f"$N={N}$, "
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
    / "polymer_energy_convergence.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
# 4. Distribución de longitudes
# ===================================

plt.figure(
    figsize=(9, 5.5)
)


plt.hist(
    bond_lengths.flatten(),
    bins=60,
    density=True,
    alpha=0.55,
    label="Simulación"
)


# Distribución teórica completa
plt.plot(
    l_grid,
    bond_probability_theory,
    linewidth=2.0,
    label=(
        r"Teoría "
        r"$P(l)\propto l^2e^{-U(l)/k_BT}$"
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
    / "polymer_bond_lengths.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ===================================
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
    f"<Ekin> experimental = {mean_kinetic:.6f}\n"
    f"<Ekin> theoretical = {kinetic_theory:.6f}\n"
    f"<Ekin>/N experimental = {mean_kinetic/N:.6f}\n"
    f"<Ekin>/N theoretical = {kinetic_theory/N:.6f}\n"
    f"relative error = {kinetic_error:.3f} %\n"
    "\n"
    f"<Eelastic> experimental = {mean_elastic:.6f}\n"
    f"<Eelastic> theoretical = {elastic_theory:.6f}\n"
    f"<Eelastic>/N experimental = {mean_elastic/N:.6f}\n"
    f"<Eelastic>/N theoretical = {elastic_theory/N:.6f}\n"
    f"relative error = {elastic_error:.3f} %\n"
    "\n"
    f"<bond length> experimental = "
    f"{mean_bond_length:.6f}\n"
    f"<bond length> theoretical = "
    f"{bond_length_theory:.6f}\n"
    f"relative error = {bond_length_error:.3f} %\n"
    f"std bond length = {std_bond_length:.6f}\n"
)


summary_file.write_text(
    summary_text
)


print()
print("===================================")
print("POLYMER RESULTS")
print("===================================")

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
    f"  Experimental = {mean_kinetic:.6f}"
)

print(
    f"  Teorica      = {kinetic_theory:.6f}"
)

print(
    f"  Experimental / N = {mean_kinetic/N:.6f}"
)

print(
    f"  Teorica / N      = {kinetic_theory/N:.6f}"
)

print(
    f"  Error = {kinetic_error:.3f} %"
)

print()

print(
    "Energia elastica:"
)

print(
    f"  Experimental = {mean_elastic:.6f}"
)

print(
    f"  Teorica      = {elastic_theory:.6f}"
)

print(
    f"  Experimental / N = {mean_elastic/N:.6f}"
)

print(
    f"  Teorica / N      = {elastic_theory/N:.6f}"
)

print(
    f"  Error = {elastic_error:.3f} %"
)

print()

print(
    "Longitud de enlace:"
)

print(
    f"  Experimental = {mean_bond_length:.6f}"
)

print(
    f"  Teorica      = {bond_length_theory:.6f}"
)

print(
    f"  Error        = {bond_length_error:.3f} %"
)

print(
    f"  std(l)       = {std_bond_length:.6f}"
)

print()

print(
    "Figures saved in:"
)

print(
    figures_folder
)