from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

        name, value = line.strip().split(
            "=",
            maxsplit=1
        )

        parameters[name] = float(value)


gamma = parameters["gamma"]
dt = parameters["dt"]

mass = parameters["mass"]
kBT = parameters["kBT"]
k = parameters["k"]

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


figures_folder = (
    project_root
    / "results"
    / "harmonic"
    / "single_run"
    / "figures"
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


file_suffix = (
    f"_{algorithm_name}"
    f"_gamma{gamma:.1f}"
    f"_h{dt:.4f}"
)


# Descartamos el primer 20 %
t_term = 0.2 * t[-1]

mask = t >= t_term


x_eq = x[mask]
v_eq = v[mask]

E_c_eq = E_c[mask]
E_p_eq = E_p[mask]


mean_Ec = np.mean(E_c_eq)
mean_Ep = np.mean(E_p_eq)

mean_Etot = np.mean(
    E_c_eq + E_p_eq
)


mean_x = np.mean(x_eq)
std_x = np.std(x_eq)

mean_v = np.mean(v_eq)
std_v = np.std(v_eq)


mean_Ec_theory = 0.5 * kBT
mean_Ep_theory = 0.5 * kBT

mean_Etot_theory = kBT

std_x_theory = np.sqrt(
    kBT / k
)

std_v_theory = np.sqrt(
    kBT / mass
)


# Energías instantáneas
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
    linestyle="--",
    label="Equipartición"
)

plt.xlabel("Tiempo $t$")
plt.ylabel("Energía")

plt.title(
    f"Evolución temporal de las energías instantáneas "
    f"({algorithm_label}, "
    f"$\\gamma={gamma:.1f}$, "
    f"$h={dt:.4f}$)"
)

plt.legend(loc="upper right")
plt.grid(True)


text_fig1 = (
    f"Promedios ($t \\geq {t_term:.1f}$):\n"
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
    figures_folder
    / f"en_inst{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)


# Convergencia de las energías
plt.figure(figsize=(8, 5))

cum_Ec = (
    np.cumsum(E_c)
    / (np.arange(len(E_c)) + 1)
)

cum_Ep = (
    np.cumsum(E_p)
    / (np.arange(len(E_p)) + 1)
)

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

plt.axhline(
    mean_Ec_theory,
    linestyle="--",
    label="Valor teórico"
)

plt.xlabel("Tiempo $t$")
plt.ylabel("Energía media acumulada")

plt.title(
    f"Convergencia del promedio temporal de energía "
    f"({algorithm_label}, "
    f"$\\gamma={gamma:.1f}$, "
    f"$h={dt:.4f}$)"
)

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    figures_folder
    / f"en_conv{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)


# Distribución de posiciones
plt.figure(figsize=(8, 5))

plt.hist(
    x_eq,
    bins=60,
    density=True,
    alpha=0.6,
    label="Simulación"
)

x_grid = np.linspace(
    min(x_eq),
    max(x_eq),
    200
)

P_x_theory = (
    1.0
    / (
        np.sqrt(2.0 * np.pi)
        * std_x_theory
    )
    * np.exp(
        -(x_grid ** 2)
        / (2.0 * std_x_theory ** 2)
    )
)

plt.plot(
    x_grid,
    P_x_theory,
    label="Gaussiana teórica"
)

plt.xlabel("Posición $x$")
plt.ylabel("Densidad de probabilidad $P(x)$")

plt.title(
    f"Distribución de posiciones $P(x)$ "
    f"({algorithm_label}, "
    f"$\\gamma={gamma:.1f}$, "
    f"$h={dt:.4f}$)"
)

plt.legend(loc="upper right")
plt.grid(True)


text_fig3 = (
    f"Estadística de $x$ "
    f"($t \\geq {t_term:.1f}$):\n"
    f"$\\mu_x$ = {mean_x:.4f}  "
    f"(Teórico: 0.0000)\n"
    f"$\\sigma_x$ = {std_x:.4f}  "
    f"(Teórico: {std_x_theory:.4f})"
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
    figures_folder
    / f"dist_pos{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)


# Distribución de velocidades
plt.figure(figsize=(8, 5))

plt.hist(
    v_eq,
    bins=60,
    density=True,
    alpha=0.6,
    label="Simulación"
)

v_grid = np.linspace(
    min(v_eq),
    max(v_eq),
    200
)

P_v_theory = (
    1.0
    / (
        np.sqrt(2.0 * np.pi)
        * std_v_theory
    )
    * np.exp(
        -(v_grid ** 2)
        / (2.0 * std_v_theory ** 2)
    )
)

plt.plot(
    v_grid,
    P_v_theory,
    label="Gaussiana teórica"
)

plt.xlabel("Velocidad $v$")
plt.ylabel("Densidad de probabilidad $P(v)$")

plt.title(
    f"Distribución de velocidades $P(v)$ "
    f"({algorithm_label}, "
    f"$\\gamma={gamma:.1f}$, "
    f"$h={dt:.4f}$)"
)

plt.legend(loc="upper right")
plt.grid(True)


text_fig4 = (
    f"Estadística de $v$ "
    f"($t \\geq {t_term:.1f}$):\n"
    f"$\\mu_v$ = {mean_v:.4f}  "
    f"(Teórico: 0.0000)\n"
    f"$\\sigma_v$ = {std_v:.4f}  "
    f"(Teórico: {std_v_theory:.4f})"
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
    figures_folder
    / f"dist_vel{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)


# Trayectoria
plt.figure(figsize=(8, 5))

plt.plot(
    t,
    x
)

plt.xlabel("Tiempo $t$")
plt.ylabel("Posición $x$")

plt.title(
    f"Trayectoria de la partícula "
    f"({algorithm_label}, "
    f"$\\gamma={gamma:.1f}$, "
    f"$h={dt:.4f}$)"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    figures_folder
    / f"trayectoria{file_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)


print()
print("===================================")
print("RESULTADOS DEL OSCILADOR ARMONICO")
print("===================================")

print(f"Algoritmo: {algorithm_label}")
print(f"gamma = {gamma}")
print(f"h = {dt}")

print()

print(
    f"<E_c> = {mean_Ec:.4f} "
    f"(teórico {mean_Ec_theory:.4f})"
)

print(
    f"<E_p> = {mean_Ep:.4f} "
    f"(teórico {mean_Ep_theory:.4f})"
)

print(
    f"<E_total> = {mean_Etot:.4f} "
    f"(teórico {mean_Etot_theory:.4f})"
)

print()

print(
    f"<x> = {mean_x:.4f}, "
    f"sigma_x = {std_x:.4f} "
    f"(teórico {std_x_theory:.4f})"
)

print(
    f"<v> = {mean_v:.4f}, "
    f"sigma_v = {std_v:.4f} "
    f"(teórico {std_v_theory:.4f})"
)

print()

print(
    "Las 5 gráficas se han guardado en:"
)

print(figures_folder)

plt.show()