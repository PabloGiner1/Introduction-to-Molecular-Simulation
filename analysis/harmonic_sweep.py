from pathlib import Path

import csv
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]

config_file = (
    project_root
    / "config"
    / "simulation.txt"
)

data_folder = (
    project_root
    / "data"
    / "raw"
    / "harmonic"
)

results_folder = (
    project_root
    / "results"
    / "harmonic"
    / "time_convergence"
)

tables_folder = (
    results_folder
    / "tables"
)

figures_folder = (
    results_folder
    / "figures"
)

tables_folder.mkdir(
    parents=True,
    exist_ok=True
)

figures_folder.mkdir(
    parents=True,
    exist_ok=True
)


results_file = (
    tables_folder
    / "harmonic_time_sweep.csv"
)

summary_file = (
    tables_folder
    / "harmonic_time_summary.csv"
)

selection_file = (
    tables_folder
    / "harmonic_time_selection.csv"
)


algorithms = {
    "EULER_MARUYAMA": "euler",
    "RUNGE_KUTTA": "rk",
    "GJF": "gjf"
}

algorithm_labels = {
    "EULER_MARUYAMA": "Euler-Maruyama",
    "RUNGE_KUTTA": "Runge-Kutta",
    "GJF": "G-JF"
}


eta_values = [
    0.1,
    1.0,
    10.0
]

simulation_times = [
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0
]

seeds = [
    12345,
    23456,
    34567,
    45678,
    56789
]

h = 0.001

theory_tolerance = 0.02
mean_tolerance = 0.05


if os.name == "nt":
    simulation_program = (
        project_root
        / "bin"
        / "simulation.exe"
    )
else:
    simulation_program = (
        project_root
        / "bin"
        / "simulation"
    )


if not simulation_program.exists():
    raise RuntimeError(
        "No se encuentra el programa compilado. "
        "Compila primero la simulacion desde VS Code."
    )


original_config = (
    config_file.read_text()
)


def write_config(
    algorithm,
    eta,
    total_time,
    seed
):

    text = (
        "# Algoritmo de integración\n"
        "# Opciones: EULER_MARUYAMA, RUNGE_KUTTA, GJF\n"
        f"algorithm={algorithm}\n"
        "\n"
        "# Coeficiente de amortiguamiento\n"
        f"eta={eta}\n"
        "\n"
        "# Paso temporal\n"
        f"h={h}\n"
        "\n"
        "# Tiempo total de simulación\n"
        f"totalTime={total_time}\n"
        "\n"
        "# Semilla del generador aleatorio\n"
        f"seed={seed}\n"
    )

    config_file.write_text(text)


def run_simulation(
    algorithm,
    eta,
    total_time,
    seed
):

    write_config(
        algorithm,
        eta,
        total_time,
        seed
    )

    result = subprocess.run(
        [str(simulation_program)],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

        raise RuntimeError(
            f"Error en la simulacion: "
            f"{algorithm}, eta={eta}, "
            f"T={total_time}, seed={seed}"
        )


def analyse_simulation(algorithm):

    algorithm_name = (
        algorithms[algorithm]
    )

    data_file = (
        data_folder
        / f"harmonic_{algorithm_name}.csv"
    )

    data = np.loadtxt(
        data_file,
        delimiter=",",
        skiprows=1
    )

    data = np.atleast_2d(data)


    if len(data) < 2:
        return None


    if not np.all(
        np.isfinite(data)
    ):
        return None


    t = data[:, 0]
    x = data[:, 1]
    v = data[:, 2]

    E_c = data[:, 3]
    E_p = data[:, 4]


    t_term = 0.2 * t[-1]

    mask = t >= t_term


    x = x[mask]
    v = v[mask]

    E_c = E_c[mask]
    E_p = E_p[mask]


    return {
        "Ec": np.mean(E_c),
        "Ep": np.mean(E_p),
        "mean_x": np.mean(x),
        "std_x": np.std(x),
        "mean_v": np.mean(v),
        "std_v": np.std(v)
    }


def calculate_average(rows):

    result = {}

    for variable in [
        "Ec",
        "Ep",
        "mean_x",
        "std_x",
        "mean_v",
        "std_v"
    ]:

        result[variable] = np.mean([
            row[variable]
            for row in rows
        ])

    return result


def calculate_maximum_error(values):

    error_Ec = (
        abs(values["Ec"] - 0.5)
        / 0.5
        * 100.0
    )

    error_Ep = (
        abs(values["Ep"] - 0.5)
        / 0.5
        * 100.0
    )

    error_std_x = (
        abs(values["std_x"] - 1.0)
        * 100.0
    )

    error_std_v = (
        abs(values["std_v"] - 1.0)
        * 100.0
    )

    return max(
        error_Ec,
        error_Ep,
        error_std_x,
        error_std_v
    )


def theory_is_correct(values):

    maximum_error = (
        calculate_maximum_error(
            values
        )
    )

    means_correct = (
        abs(values["mean_x"])
        < mean_tolerance
        and abs(values["mean_v"])
        < mean_tolerance
    )

    return (
        maximum_error
        < 100.0 * theory_tolerance
        and means_correct
    )


all_results = []
summary = []
selected_times = {}


print()
print("===================================")
print("ESTUDIO DE CONVERGENCIA TEMPORAL")
print("===================================")
print(f"h = {h}")
print(f"Semillas = {len(seeds)}")
print()


try:

    for eta in eta_values:

        print()
        print(f"eta = {eta}")
        print()

        selected_times[eta] = {}


        for algorithm in algorithms:

            print(
                algorithm_labels[algorithm]
            )

            selected_time = None


            for total_time in simulation_times:

                current_rows = []


                print(
                    f"  T = {total_time:g}",
                    end="",
                    flush=True
                )


                for seed in seeds:

                    run_simulation(
                        algorithm,
                        eta,
                        total_time,
                        seed
                    )

                    values = (
                        analyse_simulation(
                            algorithm
                        )
                    )


                    if values is None:
                        continue


                    row = {
                        "algorithm": algorithm,
                        "eta": eta,
                        "h": h,
                        "totalTime": total_time,
                        "seed": seed,
                        **values
                    }

                    all_results.append(row)
                    current_rows.append(row)


                if len(current_rows) == 0:

                    print("  ERROR")
                    continue


                average = (
                    calculate_average(
                        current_rows
                    )
                )

                maximum_error = (
                    calculate_maximum_error(
                        average
                    )
                )

                correct = (
                    theory_is_correct(
                        average
                    )
                )


                summary.append({
                    "algorithm": algorithm,
                    "eta": eta,
                    "totalTime": total_time,
                    **average,
                    "maximum_error": maximum_error,
                    "correct": int(correct)
                })


                if correct:
                    state = "OK"
                else:
                    state = "NO"


                print(
                    f"   error máximo = "
                    f"{maximum_error:6.2f} %   "
                    f"<x> = {average['mean_x']:7.4f}   "
                    f"<v> = {average['mean_v']:7.4f}   "
                    f"{state}"
                )


                if correct:

                    selected_time = (
                        total_time
                    )

                    break


            selected_times[eta][algorithm] = (
                selected_time
            )


            if selected_time is None:

                print(
                    "  No se ha alcanzado "
                    "convergencia"
                )

            else:

                print(
                    "  Tiempo mínimo convergido: "
                    f"{selected_time:g}"
                )


            print()


finally:

    config_file.write_text(
        original_config
    )


with results_file.open(
    "w",
    newline=""
) as file:

    names = [
        "algorithm",
        "eta",
        "h",
        "totalTime",
        "seed",
        "Ec",
        "Ep",
        "mean_x",
        "std_x",
        "mean_v",
        "std_v"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=names
    )

    writer.writeheader()
    writer.writerows(all_results)


with summary_file.open(
    "w",
    newline=""
) as file:

    names = [
        "algorithm",
        "eta",
        "totalTime",
        "Ec",
        "Ep",
        "mean_x",
        "std_x",
        "mean_v",
        "std_v",
        "maximum_error",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=names
    )

    writer.writeheader()
    writer.writerows(summary)


selection_rows = []


print()
print("===================================")
print("TIEMPOS SELECCIONADOS")
print("===================================")


for eta in eta_values:

    algorithm_times = (
        selected_times[eta]
    )

    valid_times = [
        time
        for time
        in algorithm_times.values()
        if time is not None
    ]

    converged = (
        len(valid_times)
        == len(algorithms)
    )


    if converged:

        selected_time = max(
            valid_times
        )

    else:

        selected_time = (
            simulation_times[-1]
        )


    selection_rows.append({
        "eta": eta,
        "selectedTime": selected_time,
        "converged": int(converged),
        "eulerTime":
            algorithm_times["EULER_MARUYAMA"],
        "rkTime":
            algorithm_times["RUNGE_KUTTA"],
        "gjfTime":
            algorithm_times["GJF"]
    })


    print()
    print(f"eta = {eta}")

    print(
        "  Euler-Maruyama: "
        f"{algorithm_times['EULER_MARUYAMA']}"
    )

    print(
        "  Runge-Kutta: "
        f"{algorithm_times['RUNGE_KUTTA']}"
    )

    print(
        "  G-JF: "
        f"{algorithm_times['GJF']}"
    )


    if converged:

        print(
            "  Tiempo seleccionado: "
            f"{selected_time:g}"
        )

    else:

        print(
            "  No han convergido todos "
            "los algoritmos"
        )


with selection_file.open(
    "w",
    newline=""
) as file:

    names = [
        "eta",
        "selectedTime",
        "converged",
        "eulerTime",
        "rkTime",
        "gjfTime"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=names
    )

    writer.writeheader()
    writer.writerows(selection_rows)


for eta in eta_values:

    plt.figure(figsize=(8, 5))


    for algorithm in algorithms:

        selected = [
            row
            for row in summary
            if (
                row["algorithm"] == algorithm
                and row["eta"] == eta
            )
        ]

        selected.sort(
            key=lambda row:
            row["totalTime"]
        )

        times = [
            row["totalTime"]
            for row in selected
        ]

        errors = [
            row["maximum_error"]
            for row in selected
        ]


        plt.plot(
            times,
            errors,
            marker="o",
            label=algorithm_labels[algorithm]
        )


    plt.axhline(
        100.0 * theory_tolerance,
        linestyle="--",
        label="Tolerancia"
    )

    plt.xscale("log")

    plt.xlabel(
        "Tiempo total de simulación"
    )

    plt.ylabel(
        "Error máximo (%)"
    )

    plt.title(
        "Convergencia con el tiempo "
        f"($\\eta={eta}$, $h={h}$)"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()


    eta_name = (
        str(eta)
        .replace(".", "p")
    )


    plt.savefig(
        figures_folder
        / f"harmonic_time_eta_{eta_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print()
print("Resultados guardados en:")
print(results_file)

print()
print("Resumen guardado en:")
print(summary_file)

print()
print("Tiempos seleccionados guardados en:")
print(selection_file)

print()
print("Gráficas guardadas en:")
print(figures_folder)