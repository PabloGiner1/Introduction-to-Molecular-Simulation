import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys


project_root = Path(__file__).resolve().parents[1]
config_file = project_root / "config" / "simulation.txt"
parameters_file = project_root / "data" / "raw" / "harmonic" / "parameters.txt"
analysis_script = project_root / "analysis" / "harmonic_analysis.py"
sweep_figures_folder = (
    project_root
    / "results"
    / "double_well"
    / "parameter_sweep"
    / "sweep_runs"
)

a_values = [0.5, 1.0, 2.0]
eta_values = [0.01, 0.1, 1.0, 2.0, 5.0]
initial_positions = [-1.0, 0.0, 1.0]

algorithms = {
    "EULER_MARUYAMA": "euler",
    "RUNGE_KUTTA": "rk",
    "GJF": "gjf",
}

simulation_program = project_root / "bin" / (
    "simulation.exe" if os.name == "nt" else "simulation"
)
if not simulation_program.exists():
    raise RuntimeError("No se encuentra el ejecutable; compila primero la simulacion.")
if not analysis_script.exists():
    raise RuntimeError(f"No se encuentra el script de analisis: {analysis_script}")


def parse_key_values(text):
    values = {}
    for line in text.splitlines():
        content = line.split("#", maxsplit=1)[0].strip()
        if "=" in content:
            name, value = content.split("=", maxsplit=1)
            values[name.strip()] = value.strip()
    return values


original_config = config_file.read_text()
base_config = parse_key_values(original_config)
algorithm = base_config.get("algorithm", "GJF")
if algorithm not in algorithms:
    raise ValueError(f"Algoritmo desconocido en simulation.txt: {algorithm}")

dt = float(base_config.get("h", "0.001"))
total_time = float(base_config.get("totalTime", "1000.0"))
seed_setting = base_config.get("seed", "random")
if dt <= 0.0 or total_time <= 0.0:
    raise ValueError("h y totalTime deben ser positivos.")

max_base_seed = (2**32 - 1) - (len(initial_positions) - 1) * 11111
if seed_setting.lower() == "random":
    base_seed = secrets.randbelow(max_base_seed + 1)
else:
    base_seed = int(seed_setting)
    if not 0 <= base_seed <= max_base_seed:
        raise ValueError(
            f"La semilla base debe estar entre 0 y {max_base_seed} "
            "para dejar espacio a las semillas de las posiciones iniciales."
        )

sweep_figures_folder.parent.mkdir(parents=True, exist_ok=True)
if sweep_figures_folder.exists():
    shutil.rmtree(sweep_figures_folder)
sweep_figures_folder.mkdir(parents=True, exist_ok=True)

completed_runs = 0
try:
    for A in a_values:
        for eta in eta_values:
            for initial_index, x0 in enumerate(initial_positions):
                seed = base_seed + initial_index * 11111
                config_file.write_text(
                    f"algorithm={algorithm}\n"
                    f"eta={eta:.12g}\n"
                    f"A={A:.12g}\n"
                    f"h={dt:.12g}\n"
                    f"totalTime={total_time:.12g}\n"
                    f"seed={seed}\n"
                    f"x0={x0:.12g}\n"
                )
                print(f"Simulando A={A:g}, eta={eta:g}, x0={x0:g}, seed={seed}")

                result = subprocess.run(
                    [str(simulation_program)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Fallo de simulacion para A={A:g}, eta={eta:g}, x0={x0:g}:\n"
                        f"{result.stdout}\n{result.stderr}"
                    )

                parameters = parse_key_values(parameters_file.read_text())
                if int(float(parameters["model"])) != 2:
                    raise RuntimeError(
                        "El ejecutable debe estar compilado para DOUBLE_WELL "
                        f"(model=2), pero uso model={parameters['model']}."
                    )
                if not (
                    abs(float(parameters["A"]) - A) < 1e-9
                    and abs(float(parameters["gamma"]) - eta) < 1e-9
                    and abs(float(parameters["x0"]) - x0) < 1e-9
                ):
                    raise RuntimeError(
                        "Los parametros escritos por la simulacion no coinciden "
                        f"con A={A:g}, eta={eta:g}, x0={x0:g}."
                    )

                analysis_environment = os.environ.copy()
                analysis_environment["MPLBACKEND"] = "Agg"
                analysis_environment["SIMULATION_ANALYSIS_OUTPUT_ROOT"] = str(
                    sweep_figures_folder
                )
                analysis_result = subprocess.run(
                    [sys.executable, str(analysis_script)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    env=analysis_environment,
                )
                if analysis_result.returncode != 0:
                    raise RuntimeError(
                        f"Fallo el analisis para A={A:g}, eta={eta:g}, x0={x0:g}:\n"
                        f"{analysis_result.stdout}\n{analysis_result.stderr}"
                    )
                completed_runs += 1
finally:
    config_file.write_text(original_config)

print(f"Barrido detallado completado: {completed_runs} simulaciones.")
