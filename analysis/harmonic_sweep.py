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

selection_summary_file = (
    tables_folder
    / "harmonic_time_selection_summary.csv"
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


# Paso temporal fijo
h = 0.001


# Criterios de validez
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


    # Descartamos el primer 20 %
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
        and
        abs(values["mean_v"])
        < mean_tolerance
    )


    return (
        maximum_error
        < 100.0 * theory_tolerance
        and means_correct
    )


def format_time(value):

    if value is None:
        return "-"

    return f"{value:g}"


all_results = []
summary = []

selected_times = {}
first_correct_times = {}


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


        selected_times[eta] = None

        first_correct_times[eta] = {
            "EULER_MARUYAMA": None,
            "RUNGE_KUTTA": None,
            "GJF": None
        }


        for total_time in simulation_times:

            print(
                f"T = {total_time:g}"
            )


            correct_at_this_time = {}


            for algorithm in algorithms:

                current_rows = []


                print(
                    f"  {algorithm_labels[algorithm]:17s}",
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

                    correct_at_this_time[
                        algorithm
                    ] = False

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


                correct_at_this_time[
                    algorithm
                ] = correct


                summary.append({
                    "algorithm": algorithm,
                    "eta": eta,
                    "totalTime": total_time,
                    **average,
                    "maximum_error": maximum_error,
                    "correct": int(correct)
                })


                if (
                    correct
                    and first_correct_times[
                        eta
                    ][algorithm] is None
                ):

                    first_correct_times[
                        eta
                    ][algorithm] = total_time


                if correct:
                    state = "OK"
                else:
                    state = "NO"


                print(
                    f"  error máximo = "
                    f"{maximum_error:6.2f} %   "
                    f"<x> = {average['mean_x']:7.4f}   "
                    f"<v> = {average['mean_v']:7.4f}   "
                    f"{state}"
                )


            # Primer tiempo en el que los tres
            # integradores cumplen el criterio
            all_correct = all(
                correct_at_this_time.get(
                    algorithm,
                    False
                )
                for algorithm in algorithms
            )


            if all_correct:

                selected_times[eta] = (
                    total_time
                )

                print()
                print(
                    "  Los tres integradores "
                    "cumplen el criterio."
                )

                print(
                    "  Tiempo seleccionado: "
                    f"{total_time:g}"
                )

                break


            print()


finally:

    config_file.write_text(
        original_config
    )


# Resultados individuales
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


# Resumen por algoritmo
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


# Tiempos seleccionados
selection_rows = []


print()
print("===================================")
print("TIEMPOS SELECCIONADOS")
print("===================================")


for eta in eta_values:

    selected_time = (
        selected_times[eta]
    )


    if selected_time is None:

        converged = 0

        selected_time = (
            simulation_times[-1]
        )

    else:

        converged = 1


    selection_rows.append({
        "eta": eta,
        "selectedTime": selected_time,
        "converged": converged,
        "eulerTime":
            first_correct_times[
                eta
            ]["EULER_MARUYAMA"],
        "rkTime":
            first_correct_times[
                eta
            ]["RUNGE_KUTTA"],
        "gjfTime":
            first_correct_times[
                eta
            ]["GJF"]
    })


    print()
    print(f"eta = {eta}")

    print(
        "  Euler-Maruyama: "
        f"{format_time(first_correct_times[eta]['EULER_MARUYAMA'])}"
    )

    print(
        "  Runge-Kutta: "
        f"{format_time(first_correct_times[eta]['RUNGE_KUTTA'])}"
    )

    print(
        "  G-JF: "
        f"{format_time(first_correct_times[eta]['GJF'])}"
    )


    if converged == 1:

        print(
            "  Tiempo común seleccionado: "
            f"{selected_time:g}"
        )

    else:

        print(
            "  No se ha encontrado un "
            "tiempo común convergido"
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


# Tabla resumen de selección temporal
time_table_rows = []


for row in selection_rows:

    eta = row["eta"]
    selected_time = row["selectedTime"]


    errors = [
        item["maximum_error"]
        for item in summary
        if (
            item["eta"] == eta
            and item["totalTime"]
            == selected_time
        )
    ]


    if len(errors) > 0:

        worst_error = max(
            errors
        )

    else:

        worst_error = np.nan


    time_table_rows.append({
        "eta": eta,
        "selectedTime": selected_time,
        "maximum_error": worst_error,
        "eulerTime": row["eulerTime"],
        "rkTime": row["rkTime"],
        "gjfTime": row["gjfTime"]
    })


with selection_summary_file.open(
    "w",
    newline=""
) as file:

    names = [
        "eta",
        "selectedTime",
        "maximum_error",
        "eulerTime",
        "rkTime",
        "gjfTime"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=names
    )

    writer.writeheader()
    writer.writerows(time_table_rows)


# Imagen de la tabla temporal
table_data = []


for row in time_table_rows:

    table_data.append([
        f"{row['eta']:g}",
        f"{row['selectedTime']:g}",
        f"{row['maximum_error']:.2f} %",
        format_time(row["eulerTime"]),
        format_time(row["rkTime"]),
        format_time(row["gjfTime"])
    ])


fig, ax = plt.subplots(
    figsize=(9, 2.5)
)

ax.axis("off")


table = ax.table(
    cellText=table_data,
    colLabels=[
        r"$\eta$",
        "T elegido",
        "Peor error",
        "Euler",
        "RK",
        "G-JF"
    ],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)


plt.title(
    "Resumen de la selección temporal",
    pad=15
)

plt.tight_layout()


plt.savefig(
    figures_folder
    / "harmonic_time_selection_table.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Gráficas detalladas para cada eta
for eta in eta_values:

    plt.figure(
        figsize=(8, 5)
    )


    for algorithm in algorithms:

        selected = [
            row
            for row in summary
            if (
                row["algorithm"]
                == algorithm
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
            label=algorithm_labels[
                algorithm
            ]
        )


    plt.axhline(
        100.0 * theory_tolerance,
        linestyle="--",
        label="Tolerancia del 2 %"
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


# Gráfica resumen de selección temporal
plt.figure(
    figsize=(8, 5)
)


for eta in eta_values:

    times = sorted(
        set(
            row["totalTime"]
            for row in summary
            if row["eta"] == eta
        )
    )


    worst_errors = []


    for total_time in times:

        errors = [
            row["maximum_error"]
            for row in summary
            if (
                row["eta"] == eta
                and row["totalTime"]
                == total_time
            )
        ]


        worst_errors.append(
            max(errors)
        )


    line, = plt.plot(
        times,
        worst_errors,
        marker="o",
        label=f"$\\eta={eta}$"
    )


    selected_time = (
        selected_times[eta]
    )


    if selected_time is not None:

        selected_errors = [
            row["maximum_error"]
            for row in summary
            if (
                row["eta"] == eta
                and row["totalTime"]
                == selected_time
            )
        ]


        selected_error = max(
            selected_errors
        )


        plt.scatter(
            selected_time,
            selected_error,
            s=180,
            marker="*",
            color=line.get_color(),
            zorder=5
        )


plt.axhline(
    100.0 * theory_tolerance,
    linestyle="--",
    label="Tolerancia del 2 %"
)


plt.xscale("log")


plt.xlabel(
    "Tiempo total de simulación"
)

plt.ylabel(
    "Peor error entre los integradores (%)"
)


plt.title(
    "Selección del tiempo de simulación"
)


plt.legend()
plt.grid(True)

plt.tight_layout()


plt.savefig(
    figures_folder
    / "harmonic_time_selection.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print()
print("Resultados individuales guardados en:")
print(results_file)

print()
print("Resumen guardado en:")
print(summary_file)

print()
print(
    "Tiempos seleccionados guardados en:"
)
print(selection_file)

print()
print(
    "Tabla resumen guardada en:"
)
print(selection_summary_file)

print()
print("Gráficas guardadas en:")
print(figures_folder)