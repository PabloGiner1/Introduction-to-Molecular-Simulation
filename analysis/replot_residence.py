# Rehace las figuras residence_distribution_A_*.png a partir de residence_times.csv,
# sin volver a lanzar las simulaciones.
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
tables = root / "results" / "double_well" / "residence_time" / "tables"
figures = root / "results" / "double_well" / "residence_time" / "figures"

eta, kBT = 1.0, 0.2      # los mismos valores que en residence_time_sweep.py
X_MAX_EN_MU = 5.0        # eje x hasta 5 veces el tiempo medio

taus = {}
with (tables / "residence_times.csv").open() as f:
    for row in csv.DictReader(f):
        taus.setdefault(float(row["A"]), []).append(float(row["tau"]))

for A, values in sorted(taus.items()):
    values = np.array(values)
    mu = values.mean()
    x_max = X_MAX_EN_MU * mu
    grid = np.linspace(0.0, x_max, 500)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40, range=(0.0, x_max), density=True,
             alpha=0.55, label="Simulación")
    plt.plot(grid, np.exp(-grid / mu) / mu, "--", linewidth=2,
             label=rf"$p(\tau)=\mu^{{-1}}e^{{-\tau/\mu}}$, $\mu={mu:.2f}$")
    plt.xlim(0.0, x_max)
    plt.xlabel(r"Tiempo de estancia $\tau$")
    plt.ylabel("Densidad de probabilidad")
    plt.title(f"Distribución de tiempos de estancia\n$A={A:g}$, $\\eta={eta:g}$, $k_BT={kBT:g}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / f"residence_distribution_A_{A:.1f}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"A={A:g}: {len(values)} estancias, mu={mu:.2f}")
