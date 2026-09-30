from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Ruta principal del proyecto
project_root = Path(__file__).resolve().parents[1]

# Archivo generado por C++
data_file = (
    project_root
    / "data"
    / "raw"
    / "random_uniform.csv"
)

# Carpeta donde se guardarán las figuras
figures_folder = (
    project_root
    / "results"
    / "figures"
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


# =======================================================
# CARGAR LOS DATOS
# =======================================================

u = np.loadtxt(
    data_file,
    delimiter=",",
    skiprows=1
)


N = len(u)


# =======================================================
# VALORES EXPERIMENTALES
# =======================================================

mean_u = np.mean(u)

variance_u = np.var(u)

std_u = np.std(u)


# Correlación entre números consecutivos
correlation = np.corrcoef(
    u[:-1],
    u[1:]
)[0, 1]


# =======================================================
# VALORES TEÓRICOS PARA U(0,1)
# =======================================================

mean_theory = 0.5

variance_theory = 1.0 / 12.0

std_theory = np.sqrt(
    variance_theory
)


# =======================================================
# FIGURA 1: HISTOGRAMA
# =======================================================

plt.figure(figsize=(8, 5))


plt.hist(
    u,
    bins=50,
    density=True,
    alpha=0.7,
    label="rand()"
)


# Para una distribución uniforme U(0,1)
# la densidad de probabilidad teórica vale 1
plt.axhline(
    1.0,
    linestyle="--",
    label="Uniforme teórica"
)


plt.xlabel("Número aleatorio $u$")
plt.ylabel("Densidad de probabilidad")


plt.title(
    f"Distribución de rand() normalizado "
    f"($N={N}$)"
)


plt.xlim(0, 1)

plt.grid(True)
plt.legend()


text = (
    f"Media = {mean_u:.6f} "
    f"(Teórico: {mean_theory:.6f})\n"

    f"Varianza = {variance_u:.6f} "
    f"(Teórico: {variance_theory:.6f})\n"

    f"Correlación consecutiva = {correlation:.6f}"
)


plt.gca().text(
    0.03,
    0.95,
    text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)


plt.tight_layout()


plt.savefig(
    figures_folder / "rand_uniform_histogram.png",
    dpi=300,
    bbox_inches="tight"
)


# =======================================================
# FIGURA 2: CORRELACIÓN ENTRE VALORES CONSECUTIVOS
# =======================================================

plt.figure(figsize=(7, 7))


# No necesitamos representar el millón entero
# para ver si aparece alguna estructura
number_scatter = 50000

plt.scatter(
    u[:number_scatter],
    u[1:number_scatter + 1],
    s=5,
    alpha=0.4
)


plt.xlabel("$u_n$")
plt.ylabel("$u_{n+1}$")


plt.title(
    f"Correlación entre valores consecutivos de rand() "
    f"({number_scatter} pares representados)"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(True)

plt.tight_layout()


plt.savefig(
    figures_folder / "rand_uniform_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

# =======================================================
# FIGURA 3: DENSIDAD 2D DE VALORES CONSECUTIVOS
# =======================================================

plt.figure(figsize=(8, 7))

plt.hexbin(
    u[:-1],
    u[1:],
    gridsize=50,
    extent=(0, 1, 0, 1),
    cmap="viridis",
    mincnt=1
)

plt.colorbar(label="Número de pares")

plt.xlabel("$u_n$")
plt.ylabel("$u_{n+1}$")

plt.title(
    f"Densidad 2D de valores consecutivos de rand() "
    f"($N={N}$)"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    figures_folder / "rand_uniform_hexbin.png",
    dpi=300,
    bbox_inches="tight"
)

# =======================================================
# RESULTADOS EN TERMINAL
# =======================================================

print()
print("===================================")
print("ANALISIS DEL GENERADOR rand()")
print("===================================")

print(f"N = {N}")

print()

print(
    f"Media experimental = {mean_u:.8f}"
)

print(
    f"Media teorica      = {mean_theory:.8f}"
)

print()

print(
    f"Varianza experimental = {variance_u:.8f}"
)

print(
    f"Varianza teorica      = {variance_theory:.8f}"
)

print()

print(
    f"Desviacion experimental = {std_u:.8f}"
)

print(
    f"Desviacion teorica      = {std_theory:.8f}"
)

print()

print(
    f"Correlacion u_n, u_n+1 = {correlation:.8f}"
)

print()

print(
    f"Las graficas se han guardado en:\n"
    f"{figures_folder}"
)


plt.show()