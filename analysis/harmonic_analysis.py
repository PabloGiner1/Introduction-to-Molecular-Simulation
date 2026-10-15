from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]

parameters_file = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
    / "parameters.txt"
)

parameters = {}

with parameters_file.open() as file:
    for line in file:
        content = line.split("#", maxsplit=1)[0].strip()
        if "=" in content:
            name, value = content.split("=", maxsplit=1)
            parameters[name.strip()] = float(value.strip())

gamma = parameters["gamma"]
dt = parameters["dt"]

mass = parameters["mass"]
kBT = parameters["kBT"]
k = parameters["k"]
A = parameters["A"]
model = int(parameters["model"])

model_names = {
    1: "harmonic",
    2: "double_well",
    3: "polymer",
}

if model not in model_names:
    raise ValueError(f"Modelo desconocido: {model}")

model_name = model_names[model]

algorithm = int(parameters["algorithm"])

if algorithm == 1:
    algorithm_name = "euler"
    algorithm_label = "Euler-Maruyama"
elif algorithm == 2:
    algorithm_name = "rk"
    algorithm_label = "Runge-Kutta"
elif algorithm == 3:
    algorithm_name = "gjf"
    algorithm_label = "G-JF"
else:
    raise ValueError("Algoritmo desconocido")

data_file = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
    / f"harmonic_{algorithm_name}.csv"
)

data = np.loadtxt(
    data_file,
    delimiter=",",
    skiprows=1
)

data = np.atleast_2d(data)

if len(data) < 2:
    raise RuntimeError(
        "La simulacion no ha generado suficientes datos. "
        "Comprueba que el algoritmo seleccionado esta implementado."
    )

t = data[:, 0]
x = data[:, 1]
v = data[:, 2]

E_c = data[:, 3]
E_p = data[:, 4]

# simulation.cpp guarda en el CSV únicamente las muestras posteriores a la termalización.
t_term = float(t[0])

x_eq = x
v_eq = v

E_c_eq = E_c
E_p_eq = E_p

mean_Ec = float(np.mean(E_c_eq))
mean_Ep = float(np.mean(E_p_eq))
mean_Etot = float(np.mean(E_c_eq + E_p_eq))

mean_x = float(np.mean(x_eq))
std_x = float(np.std(x_eq))

mean_v = float(np.mean(v_eq))
std_v = float(np.std(v_eq))

# --- VALORES TEÓRICOS EN EQUILIBRIO ---
mean_Ec_theory = 0.5 * kBT

if model_name == "double_well":
    if A <= 0.0 or kBT <= 0.0:
        raise ValueError("A y kBT deben ser positivos para double_well")

    x_limit = np.sqrt(1.0 + np.sqrt(40.0 * kBT / A))
    x_theory_grid = np.linspace(-x_limit, x_limit, 20001)
    potential_grid = A * (x_theory_grid ** 2 - 1.0) ** 2
    canonical_weights = np.exp(-potential_grid / kBT)
    position_normalization = np.trapezoid(canonical_weights, x_theory_grid)

    mean_Ep_theory = float(
        np.trapezoid(potential_grid * canonical_weights, x_theory_grid) / position_normalization
    )
    std_x_theory = float(
        np.sqrt(
            np.trapezoid(x_theory_grid ** 2 * canonical_weights, x_theory_grid)
            / position_normalization
        )
    )
else:
    mean_Ep_theory = 0.5 * kBT
    std_x_theory = float(np.sqrt(kBT / k))

mean_Etot_theory = mean_Ec_theory + mean_Ep_theory
std_v_theory = float(np.sqrt(kBT / mass))

# --- GENERACIÓN DE FIGURAS ---
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

figures_folder = (
    project_root
    / "results"
    / model_name
    / "single_run"
    / "figures"
    / f"A_{A:g}"
    / f"eta_{gamma:.2f}"
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
)

file_suffix = (
    f"_{algorithm_name}"
    f"_gamma{gamma:.2f}"
    f"_h{dt:.4f}"
    f"_run{run_timestamp}"
)

title_parameters = (
    f"$A={A:.1f}$, $\\eta={gamma:.2f}$"
    if model_name == "double_well"
    else f"$\\eta={gamma:.2f}$, $dt={dt:.4f}$"
)

# 1. Energías instantáneas
plt.figure(figsize=(8, 5))

plt.plot(
    t,
    E_c,
    label="$E_{cin}$",
    alpha=0.6
)

plt.plot(
    t,
    E_p,
    label="$E_{pot}$",
    alpha=0.6
)

plt.axhline(
    mean_Ec_theory,
    color="black",
    linestyle="--",
    label=r"Teoría $E_{cin}$ ($k_BT/2$)"
)

plt.axvline(
    t_term,
    color="gray",
    linestyle=":",
    label=f"Inicio eq. ($t={t_term:.2f}$)"
)

plt.xlabel("Tiempo $t$")
plt.xlim(left=0.0)
plt.ylabel("Energía")

plt.title(
    f"Evolución temporal de las energías instantáneas "
    f"({algorithm_label}, {title_parameters})"
)

plt.legend(loc="upper right")
plt.grid(True)

text_fig1 = (
    f"Promedios ($t \\geq {t_term:.2f}$):\n"
    f"$\\langle E_{{cin}} \\rangle$ = {mean_Ec:.4f}  "
    f"(Teórico: {mean_Ec_theory:.4f})\n"
    f"$\\langle E_{{pot}} \\rangle$ = {mean_Ep:.4f}  "
    f"(Teórico: {mean_Ep_theory:.4f})\n"
    f"$\\langle E_{{tot}} \\rangle$ = {mean_Etot:.4f}  "
    f"(Teórico: {mean_Etot_theory:.4f})"
)

plt.gca().text(
    0.03,
    0.95,
    text_fig1,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

plt.tight_layout()

plt.savefig(
    figures_folder / f"en_inst{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)

# 2. Convergencia de las energías
plt.figure(figsize=(8, 5))

cum_Ec = np.cumsum(E_c) / (np.arange(len(E_c)) + 1)
cum_Ep = np.cumsum(E_p) / (np.arange(len(E_p)) + 1)

plt.plot(
    t,
    cum_Ec,
    label=r"$\langle E_{cin} \rangle_{acum}$"
)

plt.plot(
    t,
    cum_Ep,
    label=r"$\langle E_{pot} \rangle_{acum}$"
)

# Referencias teóricas
plt.axhline(
    mean_Ec_theory,
    color="black",
    linestyle="--",
    label=r"Teórico $E_{cin}$"
)

plt.axhline(
    mean_Ep_theory,
    color="gray",
    linestyle=":",
    label=r"Teórico $E_{pot}$"
)

plt.xlabel("Tiempo $t$")
plt.xlim(left=0.0)
plt.ylabel("Energía media acumulada")

plt.title(
    f"Convergencia del promedio temporal de energía "
    f"({algorithm_label}, {title_parameters})"
)

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    figures_folder / f"en_conv{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)

# 3. Distribución de posiciones
plt.figure(figsize=(8, 5))

plt.hist(
    x_eq,
    bins=60,
    density=True,
    alpha=0.6,
    label="Simulación"
)

x_grid = np.linspace(min(x_eq), max(x_eq), 200)

if model_name == "double_well":
    P_x_theory = (
        np.exp(-A * (x_grid ** 2 - 1.0) ** 2 / kBT)
        / position_normalization
    )
    position_theory_label = "Distribución teórica de Boltzmann"
else:
    P_x_theory = (
        1.0
        / (np.sqrt(2.0 * np.pi) * std_x_theory)
        * np.exp(-(x_grid ** 2) / (2.0 * std_x_theory ** 2))
    )
    position_theory_label = "Gaussiana teórica"

plt.plot(
    x_grid,
    P_x_theory,
    color="red",
    label=position_theory_label
)

plt.xlabel("Posición $x$")
plt.ylabel("Densidad de probabilidad $P(x)$")

plt.title(
    f"Distribución de posiciones $P(x)$ "
    f"({algorithm_label}, {title_parameters})"
)

plt.legend(loc="upper right")
plt.grid(True)

text_fig3 = (
    f"Estadística de $x$ ($t \\geq {t_term:.2f}$):\n"
    f"$\\mu_x$ = {mean_x:.4f}  (Teórico: 0.0000)\n"
    f"$\\sigma_x$ = {std_x:.4f}  (Teórico: {std_x_theory:.4f})"
)

plt.gca().text(
    0.03,
    0.95,
    text_fig3,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

plt.tight_layout()

plt.savefig(
    figures_folder / f"dist_pos{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)

# 4. Distribución de velocidades
plt.figure(figsize=(8, 5))

plt.hist(
    v_eq,
    bins=60,
    density=True,
    alpha=0.6,
    label="Simulación"
)

v_grid = np.linspace(min(v_eq), max(v_eq), 200)

P_v_theory = (
    1.0
    / (np.sqrt(2.0 * np.pi) * std_v_theory)
    * np.exp(-(v_grid ** 2) / (2.0 * std_v_theory ** 2))
)

plt.plot(
    v_grid,
    P_v_theory,
    color="red",
    label="Gaussiana teórica"
)

plt.xlabel("Velocidad $v$")
plt.ylabel("Densidad de probabilidad $P(v)$")

plt.title(
    f"Distribución de velocidades $P(v)$ "
    f"({algorithm_label}, {title_parameters})"
)

plt.legend(loc="upper right")
plt.grid(True)

text_fig4 = (
    f"Estadística de $v$ ($t \\geq {t_term:.2f}$):\n"
    f"$\\mu_v$ = {mean_v:.4f}  (Teórico: 0.0000)\n"
    f"$\\sigma_v$ = {std_v:.4f}  (Teórico: {std_v_theory:.4f})"
)

plt.gca().text(
    0.03,
    0.95,
    text_fig4,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

plt.tight_layout()

plt.savefig(
    figures_folder / f"dist_vel{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)

# 5. Trayectoria
plt.figure(figsize=(8, 5))

plt.plot(t, x, label="$x(t)$")
plt.axvline(
    t_term,
    color="gray",
    linestyle=":",
    label=f"Inicio eq. ($t={t_term:.2f}$)"
)

plt.xlabel("Tiempo $t$")
plt.xlim(left=0.0)
plt.ylabel("Posición $x$")

plt.title(
    f"Trayectoria de la partícula "
    f"({algorithm_label}, {title_parameters})"
)

plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    figures_folder / f"trayectoria{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)

print()
print("===================================")
print(f"RESULTADOS DEL MODELO: {model_name.upper()}")
print("===================================")

print(f"Algoritmo: {algorithm_label}")
print(f"gamma = {gamma}")
print(f"h = {dt}")
print(f"Primera muestra tras termalización = {t_term:.4f} s")
print(f"Muestras analizadas = {len(data)}")

print()

print(f"<E_c> = {mean_Ec:.4f} (teórico {mean_Ec_theory:.4f})")
print(f"<E_p> = {mean_Ep:.4f} (teórico {mean_Ep_theory:.4f})")
print(f"<E_total> = {mean_Etot:.4f} (teórico {mean_Etot_theory:.4f})")

print()

print(f"<x> = {mean_x:.4f}, sigma_x = {std_x:.4f} (teórico {std_x_theory:.4f})")
print(f"<v> = {mean_v:.4f}, sigma_v = {std_v:.4f} (teórico {std_v_theory:.4f})")

print()
print("Las 5 gráficas se han guardado en:")
print(figures_folder)

plt.show()