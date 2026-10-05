from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]

data_file = (
    project_root
    / "data"
    / "raw"
    / "random"
    / "random_uniform.csv"
)

figures_folder = (
    project_root
    / "results"
    / "random_test"
    / "figures"
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


u = np.loadtxt(
    data_file,
    delimiter=",",
    skiprows=1
)

N = len(u)


mean_u = np.mean(u)
variance_u = np.var(u)
std_u = np.std(u)

correlation = np.corrcoef(
    u[:-1],
    u[1:]
)[0, 1]


mean_theory = 0.5
variance_theory = 1.0 / 12.0

std_theory = np.sqrt(
    variance_theory
)


# Histograma
plt.figure(figsize=(8, 5))

plt.hist(
    u,
    bins=50,
    density=True,
    alpha=0.7,
    label="rand()"
)

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


# Correlación
plt.figure(figsize=(7, 7))

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


# Densidad 2D
plt.figure(figsize=(8, 7))

plt.hexbin(
    u[:-1],
    u[1:],
    gridsize=50,
    extent=(0, 1, 0, 1),
    cmap="viridis",
    mincnt=1
)

plt.colorbar(
    label="Número de pares"
)

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

print("Las graficas se han guardado en:")
print(figures_folder)

plt.show()