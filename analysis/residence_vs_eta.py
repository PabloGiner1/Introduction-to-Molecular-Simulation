# Tiempo medio de estancia mu frente al damping eta en el doble pozo
# V(x) = A (x^2 - 1)^2, con A fijo, comparado con la teoria de Kramers.
#
# Uso: copiar en analysis/ y ejecutar desde la raiz del proyecto:
#     python analysis/residence_vs_eta.py
# Necesita bin/simulation(.exe) ya compilado. Reescribe config/simulation.txt
# durante el barrido y lo deja como estaba al terminar (tambien si falla o se
# corta con Ctrl+C).
#
# Resultados en results/double_well/residence_time_eta/ (tablas y figuras).

from pathlib import Path
import csv
import os
import secrets
import subprocess
import time as clock

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None


# ===================================
# Parametros del estudio
# ===================================

A = 0.8            # barrera = 4 kBT (ver explicacion en el mensaje)
KBT = 0.2
MASS = 1.0
DT = 0.01
TOTAL_TIME = 100000.0
SAVED_DT = 0.2     # resolucion temporal del CSV (antes era ~0.25-1 o peor)

ETA_VALUES = np.array([0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2, 2.5, 5.0])
INITIAL_POSITIONS = [-1.0, 0.0, 1.0]   # tres replicas, como en residence_time_sweep.py
THRESHOLD = 0.5                        # criterio de umbral del repo: x = +-0.5
CAPTURE_MARGIN = 1.0                   # atrapado si E < A - CAPTURE_MARGIN*kBT
MAX_ATTEMPTS = 4                       # reintentos con otra semilla si no termaliza


# ===================================
# Rutas
# ===================================

project_root = Path(__file__).resolve().parents[1]
config_file = project_root / "config" / "simulation.txt"
data_file = project_root / "data" / "raw" / "harmonic" / "harmonic_gjf.csv"
parameters_file = project_root / "data" / "raw" / "harmonic" / "parameters.txt"

results_folder = project_root / "results" / "double_well" / "residence_time_eta"
figures_folder = results_folder / "figures"
tables_folder = results_folder / "tables"
figures_folder.mkdir(parents=True, exist_ok=True)
tables_folder.mkdir(parents=True, exist_ok=True)
data_file.parent.mkdir(parents=True, exist_ok=True)

simulation_program = project_root / "bin" / ("simulation.exe" if os.name == "nt" else "simulation")
if not simulation_program.exists():
    raise RuntimeError(f"No se encuentra {simulation_program}. Compila primero.")


# ===================================
# Teoria de Kramers para V = A (x^2 - 1)^2
# ===================================
# omega0^2 = V''(+-1)/m = 8A/m,  omegab^2 = |V''(0)|/m = 4A/m
# k_TST  = omega0/(2 pi) exp(-A/kBT)
# Friccion alta/intermedia (Kramers):  k = (lambda/omegab) k_TST,
#     lambda = sqrt(eta^2/4 + omegab^2) - eta/2
# Friccion baja (difusion en energia): k = eta I_b/kBT k_TST,
#     I_b = accion de la orbita a la energia de la barrera = (8/3) sqrt(m A)
# En el limite de friccion baja la particula que llega a la energia de la
# barrera acaba atrapada en cualquiera de los dos pozos con probabilidad 1/2,
# asi que el ritmo de saltos de pozo a pozo es k_baja/2 y mu = 2/k_baja.
# Interpolacion simple entre ambos regimenes: mu = 2/k_baja + 1/k_alta.

omega0 = np.sqrt(8.0 * A / MASS)
omegab = np.sqrt(4.0 * A / MASS)
k_tst = omega0 / (2.0 * np.pi) * np.exp(-A / KBT)
action_barrier = 8.0 / 3.0 * np.sqrt(MASS * A)


def k_high(eta):
    lam = np.sqrt(eta ** 2 / 4.0 + omegab ** 2) - eta / 2.0
    return lam / omegab * k_tst


def k_low(eta):
    return eta * action_barrier / KBT * k_tst


def mu_low(eta):
    return 2.0 / k_low(eta)


def mu_high(eta):
    return 1.0 / k_high(eta)


def mu_turnover(eta):
    return mu_low(eta) + mu_high(eta)


# ===================================
# Deteccion de estancias
# ===================================

def residences_from_states(time, state):
    # state[i] = -1 (pozo izquierdo), +1 (derecho) o 0 (indefinido: se
    # mantiene el ultimo estado). Se descartan la primera y la ultima estancia
    # porque estan cortadas por el inicio y el final de la serie.
    defined = np.flatnonzero(state != 0)
    if len(defined) == 0:
        return np.array([])
    s = state[defined]
    changes = defined[1:][s[1:] != s[:-1]]
    return np.diff(time[changes])


def threshold_residences(time, x, c=THRESHOLD):
    # Mismo criterio que residence_time_sweep.py: se cambia de pozo al llegar
    # a x = +c (desde la izquierda) o a x = -c (desde la derecha).
    state = np.where(x <= -c, -1, np.where(x >= c, 1, 0))
    return residences_from_states(time, state)


def energy_residences(time, x, energy, c=THRESHOLD):
    # Se cambia de pozo solo cuando la particula pasa el umbral x = +-c Y queda
    # ATRAPADA en el otro pozo: energia total E < A - CAPTURE_MARGIN*kBT.
    # Asi una particula con E > A que va y viene por encima de la barrera
    # (lo normal con eta pequeno) no cuenta cada oscilacion como un salto.
    # Con eta grande E queda siempre bajo la barrera y equivale al umbral.
    trapped = energy < A - CAPTURE_MARGIN * KBT
    state = np.where(trapped & (x <= -c), -1, np.where(trapped & (x >= c), 1, 0))
    return residences_from_states(time, state)


def read_trajectory():
    if pd is not None:
        data = pd.read_csv(data_file, usecols=["time", "x", "kinetic", "potential"]).to_numpy()
    else:
        data = np.loadtxt(data_file, delimiter=",", skiprows=1, usecols=(0, 1, 3, 4))
    if len(data) < 2 or not np.all(np.isfinite(data)):
        raise RuntimeError("Datos no validos en " + str(data_file))
    return data[:, 0], data[:, 1], data[:, 2] + data[:, 3]


def read_parameters():
    values = {}
    for line in parameters_file.read_text().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def mean_and_error(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan
    # error del promedio entre replicas: desviacion tipica / sqrt(n)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def save_figure(path):
    # OneDrive (o un visor abierto) puede bloquear el PNG: se guarda con otro nombre.
    try:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    except PermissionError:
        alternative = path.with_name(path.stem + "_nuevo" + path.suffix)
        plt.savefig(alternative, dpi=300, bbox_inches="tight")
        print(f"AVISO: {path.name} estaba bloqueado, guardado como {alternative.name}")


# ===================================
# Barrido
# ===================================

original_config = config_file.read_text()
base_seed = secrets.randbelow(2 ** 31 - 10 ** 7)
max_saved_points = int(round(TOTAL_TIME / SAVED_DT))

replicate_rows = []
summary_rows = []
start = clock.time()

try:
    for eta_index, eta in enumerate(ETA_VALUES):
        print(f"\n=== eta = {eta:g}  ({eta_index + 1}/{len(ETA_VALUES)}) ===")
        mu_energy_reps, mu_threshold_reps = [], []
        n_energy_total, n_threshold_total = 0, 0

        for rep, x0 in enumerate(INITIAL_POSITIONS):
            for attempt in range(MAX_ATTEMPTS):
                seed = base_seed + eta_index * 100000 + rep * 11111 + attempt * 997
                config_file.write_text(
                    "model=DOUBLE_WELL\n"
                    "algorithm=GJF\n"
                    f"eta={eta:.12g}\n"
                    f"kBT={KBT:.12g}\n"
                    f"mass={MASS:.12g}\n"
                    f"A={A:.12g}\n"
                    "force=0.0\n"
                    f"h={DT:.12g}\n"
                    f"totalTime={TOTAL_TIME:.12g}\n"
                    f"maxSavedPoints={max_saved_points}\n"
                    f"seed={seed}\n"
                    f"x0={x0:.12g}\n"
                    "v0=0.0\n"
                )
                result = subprocess.run([str(simulation_program)], cwd=project_root,
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    break
                print(f"  Rep {rep + 1}: fallo (seguramente no termalizo), reintento con otra semilla")
            else:
                raise RuntimeError(f"La simulacion falla con eta={eta:g}:\n{result.stdout}\n{result.stderr}")

            params = read_parameters()
            time, x, energy = read_trajectory()
            tau_energy = energy_residences(time, x, energy)
            tau_threshold = threshold_residences(time, x)

            mu_e = tau_energy.mean() if len(tau_energy) else np.nan
            mu_t = tau_threshold.mean() if len(tau_threshold) else np.nan
            mu_energy_reps.append(mu_e)
            mu_threshold_reps.append(mu_t)
            n_energy_total += len(tau_energy)
            n_threshold_total += len(tau_threshold)

            replicate_rows.append({
                "eta": eta, "replicate": rep + 1, "x0": x0, "seed": seed,
                "n_energy": len(tau_energy), "mu_energy": mu_e,
                "n_threshold": len(tau_threshold), "mu_threshold": mu_t,
                "thermalization_time": float(params.get("thermalizationTime", np.nan)),
                "saved_dt": float(params.get("savedDt", np.nan)),
            })
            print(f"  Rep {rep + 1} (x0={x0:g}): mu = {mu_e:8.1f} con {len(tau_energy):4d} estancias "
                  f"[umbral: mu = {mu_t:8.1f}, {len(tau_threshold)} estancias]")

        mu_e, err_e = mean_and_error(mu_energy_reps)
        mu_t, err_t = mean_and_error(mu_threshold_reps)
        summary_rows.append({
            "eta": eta,
            "mu_energy": mu_e, "mu_energy_err": err_e, "n_energy": n_energy_total,
            "mu_threshold": mu_t, "mu_threshold_err": err_t, "n_threshold": n_threshold_total,
            "mu_kramers_turnover": mu_turnover(eta),
            "mu_kramers_high": mu_high(eta),
            "mu_kramers_low": mu_low(eta),
        })
        elapsed = clock.time() - start
        remaining = elapsed / (eta_index + 1) * (len(ETA_VALUES) - eta_index - 1)
        print(f"  -> mu = {mu_e:.1f} +- {err_e:.1f}   (teoria: {mu_turnover(eta):.1f})"
              f"   [{elapsed / 60:.1f} min, quedan ~{remaining / 60:.1f} min]")

finally:
    config_file.write_text(original_config)
    print("\nconfig/simulation.txt restaurado.")


# ===================================
# Tablas
# ===================================

for name, rows in [("residence_vs_eta_replicates.csv", replicate_rows),
                   ("residence_vs_eta_summary.csv", summary_rows)]:
    with (tables_folder / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ===================================
# Figura: mu frente a eta (log-log)
# ===================================

eta_s = np.array([r["eta"] for r in summary_rows])
mu_e = np.array([r["mu_energy"] for r in summary_rows])
err_e = np.array([r["mu_energy_err"] for r in summary_rows])
mu_t = np.array([r["mu_threshold"] for r in summary_rows])
err_t = np.array([r["mu_threshold_err"] for r in summary_rows])
eta_grid = np.logspace(np.log10(eta_s.min()) - 0.2, np.log10(eta_s.max()) + 0.2, 300)

plt.figure(figsize=(8, 5.5))
plt.plot(eta_grid, mu_turnover(eta_grid), "k-", lw=2,
         label=r"Kramers: $\mu = 2/k_{\rm baja} + 1/k_{\rm alta}$")
plt.plot(eta_grid, mu_low(eta_grid), ":", color="tab:green",
         label=r"Fricción baja: $\mu = 2/k_{\rm baja} \propto 1/\eta$")
plt.plot(eta_grid, mu_high(eta_grid), "--", color="tab:purple",
         label=r"Fricción alta (Kramers): $\mu \to \propto \eta$")
plt.axhline(1.0 / k_tst, color="gray", lw=1, alpha=0.6, label=r"Estado de transición $1/k_{\rm TST}$")
plt.errorbar(eta_s, mu_e, yerr=err_e, fmt="o", ms=7, capsize=4, color="tab:blue",
             label=rf"Simulación (atrapado: $|x|\geq{THRESHOLD:g}$ y $E<A-{CAPTURE_MARGIN:g}\,k_BT$)")
plt.errorbar(eta_s, mu_t, yerr=err_t, fmt="s", ms=6, mfc="none", capsize=3, color="tab:orange",
             label=rf"Simulación (solo umbral $x=\pm{THRESHOLD:g}$, como el repo)")
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.5 * np.nanmin(np.r_[mu_e, mu_t]), 2.0 * np.nanmax(np.r_[mu_e, mu_t]))
plt.xlabel(r"Coeficiente de fricción $\eta$")
plt.ylabel(r"Tiempo medio de estancia $\mu=\langle\tau\rangle$")
plt.title("Tiempo medio de estancia frente a la fricción\n"
          rf"$A={A:g}$, $k_BT={KBT:g}$ ($A/k_BT={A / KBT:g}$), GJF, $h={DT:g}$")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8.5)
plt.tight_layout()
save_figure(figures_folder / "mean_residence_vs_eta.png")
plt.close()


# ===================================
# Resumen
# ===================================

best = int(np.nanargmin(mu_e))
print("\n eta     mu(atrapado)      mu(umbral)     Kramers   N")
for r in summary_rows:
    print(f"{r['eta']:5g}  {r['mu_energy']:7.1f} +- {r['mu_energy_err']:5.1f}  "
          f"{r['mu_threshold']:7.1f} +- {r['mu_threshold_err']:5.1f}  "
          f"{r['mu_kramers_turnover']:7.1f}  {r['n_energy']}")
print(f"\nMinimo simulado en eta = {eta_s[best]:g} (mu = {mu_e[best]:.1f})")
eta_theory_min = eta_grid[np.argmin(mu_turnover(eta_grid))]
print(f"Minimo de la interpolacion de Kramers en eta ~ {eta_theory_min:.2g}")
print(f"Tiempo total: {(clock.time() - start) / 60:.1f} min")
print("Resultados en:", results_folder)
