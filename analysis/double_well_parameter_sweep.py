from pathlib import Path

import csv
import math
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
config_file = project_root / "config" / "simulation.txt"
data_folder = project_root / "data" / "raw" / "harmonic"
parameters_file = data_folder / "parameters.txt"
results_folder = project_root / "results" / "double_well" / "parameter_sweep"
tables_folder = results_folder / "tables"
figures_folder = results_folder / "figures"

tables_folder.mkdir(parents=True, exist_ok=True)
figures_folder.mkdir(parents=True, exist_ok=True)

a_values = [0.1, 0.5, 1.0, 2.0]
eta_values = [0.1, 1.0, 10.0]
replicate_count = 3

algorithms = {
    "EULER_MARUYAMA": "euler",
    "RUNGE_KUTTA": "rk",
    "GJF": "gjf",
}

if os.name == "nt":
    simulation_program = project_root / "bin" / "simulation.exe"
else:
    simulation_program = project_root / "bin" / "simulation"

if not simulation_program.exists():
    raise RuntimeError("No se encuentra el ejecutable; compila primero la simulacion.")


def parse_key_values(text):
    values = {}
    for line in text.splitlines():
        content = line.split("#", maxsplit=1)[0].strip()
        if "=" in content:
            name, value = content.split("=", maxsplit=1)
            values[name.strip()] = value.strip()
    return values


def theoretical_potential_mean(A, kBT):
    if A <= 0.0 or kBT <= 0.0:
        raise ValueError("A y kBT deben ser positivos para double_well")

    x_limit = math.sqrt(1.0 + math.sqrt(40.0 * kBT / A))
    x_grid = np.linspace(-x_limit, x_limit, 20001)
    potential = A * (x_grid ** 2 - 1.0) ** 2
    weights = np.exp(-potential / kBT)
    partition_function = np.trapezoid(weights, x_grid)
    return np.trapezoid(potential * weights, x_grid) / partition_function


def relative_deviation(value, reference):
    return abs(value - reference) / abs(reference) * 100.0


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


original_config = config_file.read_text()
base_config = parse_key_values(original_config)
algorithm = base_config.get("algorithm", "GJF")
if algorithm not in algorithms:
    raise ValueError(f"Algoritmo desconocido en simulation.txt: {algorithm}")

dt = float(base_config.get("h", "0.001"))
total_time = float(base_config.get("totalTime", "1000.0"))
base_seed = int(base_config.get("seed", "12345"))
if dt <= 0.0 or total_time <= 0.0:
    raise ValueError("h y totalTime deben ser positivos")

simulation_results = []

try:
    for A in a_values:
        for eta in eta_values:
            for replicate in range(replicate_count):
                seed = base_seed + replicate * 11111
                config_file.write_text(
                    f"algorithm={algorithm}\n"
                    f"eta={eta:.12g}\n"
                    f"A={A:.12g}\n"
                    f"h={dt:.12g}\n"
                    f"totalTime={total_time:.12g}\n"
                    f"seed={seed}\n"
                )
                print(f"Simulando A={A:g}, eta={eta:g}, seed={seed}")

                result = subprocess.run(
                    [str(simulation_program)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Fallo para A={A:g}, eta={eta:g}, seed={seed}:\n"
                        f"{result.stdout}\n{result.stderr}"
                    )

                parameters = parse_key_values(parameters_file.read_text())
                model = int(float(parameters["model"]))
                if model != 2:
                    raise RuntimeError(
                        f"Se esperaba DOUBLE_WELL, pero el ejecutable uso model={model}."
                    )

                actual_A = float(parameters["A"])
                actual_eta = float(parameters["gamma"])
                kBT = float(parameters["kBT"])
                actual_dt = float(parameters["dt"])
                actual_total_time = float(parameters["totalTime"])
                data_file = data_folder / f"harmonic_{algorithms[algorithm]}.csv"
                data = np.atleast_2d(
                    np.loadtxt(data_file, delimiter=",", skiprows=1)
                )
                if len(data) < 2:
                    raise RuntimeError(f"Datos insuficientes para A={A:g}, eta={eta:g}.")

                equilibrium = data[:, 0] >= 0.2 * data[-1, 0]
                if not np.any(equilibrium):
                    raise RuntimeError("No hay muestras en la region de equilibrio.")

                mean_kinetic = float(np.mean(data[equilibrium, 3]))
                mean_potential = float(np.mean(data[equilibrium, 4]))
                kinetic_theory = 0.5 * kBT
                potential_theory = theoretical_potential_mean(actual_A, kBT)
                simulation_results.append({
                    "A": actual_A,
                    "eta": actual_eta,
                    "seed": seed,
                    "algorithm": algorithm,
                    "dt": actual_dt,
                    "totalTime": actual_total_time,
                    "kBT": kBT,
                    "mean_kinetic": mean_kinetic,
                    "kinetic_theory": kinetic_theory,
                    "kinetic_deviation_percent": relative_deviation(
                        mean_kinetic, kinetic_theory
                    ),
                    "mean_potential": mean_potential,
                    "potential_boltzmann_theory": potential_theory,
                    "potential_deviation_from_boltzmann_percent": relative_deviation(
                        mean_potential, potential_theory
                    ),
                    "potential_kBT_over_2_reference": kinetic_theory,
                    "potential_deviation_from_kBT_over_2_percent": relative_deviation(
                        mean_potential, kinetic_theory
                    ),
                })
finally:
    config_file.write_text(original_config)


summary = []
for A in a_values:
    for eta in eta_values:
        runs = [
            row for row in simulation_results
            if math.isclose(row["A"], A) and math.isclose(row["eta"], eta)
        ]
        kinetic_values = [row["mean_kinetic"] for row in runs]
        potential_values = [row["mean_potential"] for row in runs]
        kinetic_mean = float(np.mean(kinetic_values))
        potential_mean = float(np.mean(potential_values))
        kinetic_theory = runs[0]["kinetic_theory"]
        potential_theory = runs[0]["potential_boltzmann_theory"]
        summary.append({
            "A": A,
            "eta": eta,
            "replicates": len(runs),
            "mean_kinetic": kinetic_mean,
            "std_kinetic": float(np.std(kinetic_values, ddof=1)),
            "kinetic_theory": kinetic_theory,
            "kinetic_deviation_percent": relative_deviation(
                kinetic_mean, kinetic_theory
            ),
            "mean_potential": potential_mean,
            "std_potential": float(np.std(potential_values, ddof=1)),
            "potential_boltzmann_theory": potential_theory,
            "potential_deviation_from_boltzmann_percent": relative_deviation(
                potential_mean, potential_theory
            ),
            "potential_kBT_over_2_reference": kinetic_theory,
            "potential_deviation_from_kBT_over_2_percent": relative_deviation(
                potential_mean, kinetic_theory
            ),
        })

write_csv(tables_folder / "double_well_energy_sweep.csv", simulation_results)
write_csv(tables_folder / "double_well_energy_summary.csv", summary)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for eta in eta_values:
    rows = [row for row in summary if row["eta"] == eta]
    axes[0].errorbar(
        [row["A"] for row in rows],
        [row["mean_kinetic"] for row in rows],
        yerr=[row["std_kinetic"] for row in rows],
        marker="o",
        capsize=3,
        label=fr"Simulación, $\eta={eta:g}$",
    )
    axes[1].errorbar(
        [row["A"] for row in rows],
        [row["mean_potential"] for row in rows],
        yerr=[row["std_potential"] for row in rows],
        marker="o",
        capsize=3,
        label=fr"Simulación, $\eta={eta:g}$",
    )

theory_rows = [row for row in summary if row["eta"] == eta_values[0]]
axes[0].axhline(
    theory_rows[0]["kinetic_theory"],
    color="black",
    linestyle="--",
    label=r"Teoría: $k_BT/2$",
)
axes[1].plot(
    [row["A"] for row in theory_rows],
    [row["potential_boltzmann_theory"] for row in theory_rows],
    color="black",
    linestyle="--",
    marker="s",
    label="Teoría de Boltzmann",
)
axes[1].axhline(
    theory_rows[0]["potential_kBT_over_2_reference"],
    color="gray",
    linestyle=":",
    label=r"Referencia armónica: $k_BT/2$",
)
axes[0].set_title("Energía cinética media")
axes[1].set_title("Energía potencial media")
for axis in axes:
    axis.set_xlabel("Parámetro $A$")
    axis.set_ylabel("Energía")
    axis.grid(True, alpha=0.3)
    axis.legend()
fig.tight_layout()
fig.savefig(figures_folder / "double_well_mean_energies.png", dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for eta in eta_values:
    rows = [row for row in summary if row["eta"] == eta]
    axes[0].plot(
        [row["A"] for row in rows],
        [row["kinetic_deviation_percent"] for row in rows],
        marker="o",
        label=fr"$\eta={eta:g}$",
    )
    axes[1].plot(
        [row["A"] for row in rows],
        [row["potential_deviation_from_boltzmann_percent"] for row in rows],
        marker="o",
        label=fr"$\eta={eta:g}$",
    )
axes[0].set_title(r"Desviación de $\langle E_{cin}\rangle$ respecto a $k_BT/2")
axes[1].set_title("Desviación de $<E_{pot}>$ respecto a Boltzmann")
for axis in axes:
    axis.set_xlabel("Parámetro $A$")
    axis.set_ylabel("Desviación relativa (%)")
    axis.grid(True, alpha=0.3)
    axis.legend(title=r"Amortiguamiento $\eta$")
fig.tight_layout()
fig.savefig(figures_folder / "double_well_energy_deviations.png", dpi=300)
plt.close(fig)

print("Barrido completado.")
print(f"Tablas: {tables_folder}")
print(f"Figuras: {figures_folder}")