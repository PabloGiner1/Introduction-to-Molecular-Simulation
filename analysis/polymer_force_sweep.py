from pathlib import Path

import csv
import math
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
config_file = project_root / "config" / "simulation.txt"
data_file = project_root / "data" / "raw" / "polymer" / "polymer.csv"
results_folder = project_root / "results" / "polymer" / "force_sweep"
tables_folder = results_folder / "tables"
figures_folder = results_folder / "figures"
summary_file = tables_folder / "polymer_force_sweep.csv"
figure_file = figures_folder / "polymer_end_to_end_vs_force.png"

maximum_force = 100.0
number_of_forces = 16


def parse_config(text):
    values = {}

    for line in text.splitlines():
        content = line.split("#", maxsplit=1)[0].strip()

        if "=" in content:
            name, value = content.split("=", maxsplit=1)
            values[name.strip()] = value.strip()

    return values


def set_config_value(text, name, value):
    lines = text.splitlines()
    updated = False

    for index, line in enumerate(lines):
        content, separator, comment = line.partition("#")

        if "=" not in content:
            continue

        key, _ = content.split("=", maxsplit=1)

        if key.strip() == name:
            suffix = f"  #{comment}" if separator else ""
            lines[index] = f"{key}={value}{suffix}"
            updated = True
            break

    if not updated:
        lines.append(f"{name}={value}")

    return "\n".join(lines) + "\n"


def read_final_measurement(number_of_monomers):
    with data_file.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No hay datos en {data_file}.")

    final_row = rows[-1]
    try:
        final_time = float(final_row["time"])
        first_x = float(final_row["x0"])
        last_x = float(final_row[f"x{number_of_monomers - 1}"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "El archivo de resultados del polímero no tiene "
            "las columnas esperadas."
        ) from error

    if not all(math.isfinite(value) for value in (final_time, first_x, last_x)):
        raise RuntimeError("La posición final del polímero no es finita.")

    return final_time, last_x - first_x


def main():
    if os.name == "nt":
        simulation_program = project_root / "bin" / "simulation.exe"
    else:
        simulation_program = project_root / "bin" / "simulation"

    if not simulation_program.exists():
        raise RuntimeError("No se encuentra el ejecutable. Compila primero.")

    original_config = config_file.read_text()
    config_values = parse_config(original_config)

    if "N" not in config_values:
        raise RuntimeError("Falta el parámetro N en config/simulation.txt.")

    number_of_monomers = int(float(config_values["N"]))
    forces_low = np.linspace(1.0, 20, 19, endpoint = False)
    forces = np.linspace(20.0, maximum_force, number_of_forces)
    measurements = []

    tables_folder.mkdir(parents=True, exist_ok=True)
    figures_folder.mkdir(parents=True, exist_ok=True)

    # try:
    #     for force in np.concatenate((forces_low, forces)):
    #         run_config = set_config_value(original_config, "model", "POLYMER")
    #         run_config = set_config_value(
    #             run_config,
    #             "force",
    #             f"{force:.8g}",
    #         )
    #         config_file.write_text(run_config)

    #         print(f"Simulando fuerza {force:g}...")
    #         subprocess.run(
    #             [str(simulation_program)],
    #             cwd=project_root,
    #             check=True,
    #         )

    #         final_time, end_to_end_x = read_final_measurement(
    #             number_of_monomers
    #         )
    #         measurements.append((force, end_to_end_x, final_time))
    #         print(
    #             f"  t_final = {final_time:g}, "
    #             f"distancia en +x = {end_to_end_x:g}"
    #         )
    # finally:
    #     config_file.write_text(original_config)


    # with summary_file.open("w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(("force", "end_to_end_x", "final_time"))
    #     writer.writerows(measurements)

    # measured_forces = [row[0] for row in measurements]
    # distances = [row[1] for row in measurements]

    # Leer los datos de la tabla de resultados
    with summary_file.open(newline="") as file:
        rows = list(csv.DictReader(file))

    measured_forces = np.array([float(row["force"]) for row in rows])
    distances = np.array([float(row["end_to_end_x"]) for row in rows])

    #Funcion de langevin teorica
    N= int(float(config_values["N"]))
    l0 = float(config_values["l0"])
    kBT = float(config_values["kBT"])
    ke = float(config_values["ke"])

    

    x = np.array(measured_forces) * l0 / kBT
    Langevin = N * l0 * (1 / np.tanh(x) - 1 / x)

    
    figure_file = figures_folder / f"polymer_end_to_end_vs_force_k_{ke:g}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(measured_forces, distances, marker=".", linestyle="None", label="Datos simulados", color="blue")
    plt.xlabel("Fuerza aplicada")
    plt.ylabel("$R_{ee}(x)$")
    plt.title("Extensión del polímero en función de la fuerza\n"
              f"$N={N:g}, l_0={l0:g}, k_e={ke:g}$")
    plt.plot(measured_forces, Langevin, label="Función de Langevin", color="orange")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_file, dpi=200)
    plt.close()

    print(f"Tabla guardada en: {summary_file}")
    print(f"Gráfica guardada en: {figure_file}")


if __name__ == "__main__":
    main()
