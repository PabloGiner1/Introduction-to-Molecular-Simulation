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

time_selection_file = (
    project_root
    / "results"
    / "harmonic"
    / "time_convergence"
    / "tables"
    / "harmonic_time_selection.csv"
)

results_folder = (
    project_root
    / "results"
    / "harmonic"
    / "algorithm_comparison"
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
    / "harmonic_algorithm_sweep.csv"
)

summary_file = (
    tables_folder
    / "harmonic_algorithm_summary.csv"
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

h_values = [
    0.0001,
    0.001,
    0.01,
    0.1
]

seeds = [
    12345,
    23456,
    34567
]


theory_tolerance = 0.02
mean_tolerance = 0.05


if not time_selection_file.exists():

    raise RuntimeError(
        "No existe harmonic_time_selection.csv. "
        "Ejecuta primero el estudio "
        "de convergencia temporal."
    )


selected_times = {}


with time_selection_file.open() as file:

    reader = csv.DictReader(file)

    for row in reader:

        eta = float(
            row["eta"]
        )

        converged = int(
            row["converged"]
        )

        if converged != 1:

            raise RuntimeError(
                f"No se ha encontrado un tiempo "
                f"convergido para eta={eta}. "
                f"Revisa primero el estudio temporal."
            )

        selected_times[eta] = float(
            row["selectedTime"]
        )


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
        "Compila primero la simulacion."
    )


original_config = (
    config_file.read_text()
)


def write_config(
    algorithm,
    eta,
    h,
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
    h,
    total_time,
    seed
):

    write_config(
        algorithm,
        eta,
        h,
        total_time,
        seed
    )

    result = subprocess.run(
        [str(simulation_program)],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    return (
        result.returncode == 0
    )


def analyse_simulation(algorithm):

    algorithm_name = (
        algorithms[algorithm]
    )

    data_file = (
        data_folder
        / f"harmonic_{algorithm_name}.csv"
    )


    try:

        data = np.loadtxt(
            data_file,
            delimiter=",",
            skiprows=1
        )

    except Exception:

        return None


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


all_results = []


print()
print("===================================")
print("ESTUDIO DE ALGORITMOS")
print("===================================")
print(f"Semillas = {len(seeds)}")
print()

print("Tiempos utilizados:")

for eta in eta_values:

    print(
        f"  eta = {eta}: "
        f"T = {selected_times[eta]:g}"
    )

print()


try:

    for eta in eta_values:

        total_time = (
            selected_times[eta]
        )

        print()
        print(
            f"eta = {eta}, "
            f"T = {total_time:g}"
        )
        print()


        for algorithm in algorithms:

            print(
                algorithm_labels[algorithm]
            )


            for h in h_values:

                print(
                    f"  h = {h:g}",
                    end="",
                    flush=True
                )


                valid_runs = 0


                for seed in seeds:

                    success = (
                        run_simulation(
                            algorithm,
                            eta,
                            h,
                            total_time,
                            seed
                        )
                    )


                    if success:

                        values = (
                            analyse_simulation(
                                algorithm
                            )
                        )

                    else:

                        values = None


                    if values is None:

                        row = {
                            "algorithm": algorithm,
                            "eta": eta,
                            "h": h,
                            "totalTime": total_time,
                            "seed": seed,
                            "valid": 0,
                            "Ec": np.nan,
                            "Ep": np.nan,
                            "mean_x": np.nan,
                            "std_x": np.nan,
                            "mean_v": np.nan,
                            "std_v": np.nan
                        }

                    else:

                        valid_runs += 1

                        row = {
                            "algorithm": algorithm,
                            "eta": eta,
                            "h": h,
                            "totalTime": total_time,
                            "seed": seed,
                            "valid": 1,
                            **values
                        }


                    all_results.append(row)


                print(
                    f"  {valid_runs}/"
                    f"{len(seeds)} correctas"
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
        "valid",
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


summary = []


for eta in eta_values:

    for h in h_values:

        for algorithm in algorithms:

            selected = [
                row
                for row in all_results
                if (
                    row["algorithm"]
                    == algorithm
                    and row["eta"] == eta
                    and row["h"] == h
                    and row["valid"] == 1
                )
            ]


            valid_runs = len(selected)


            if valid_runs == 0:

                summary.append({
                    "algorithm": algorithm,
                    "eta": eta,
                    "h": h,
                    "valid_runs": 0,
                    "Ec": np.nan,
                    "Ep": np.nan,
                    "mean_x": np.nan,
                    "std_x": np.nan,
                    "mean_v": np.nan,
                    "std_v": np.nan,
                    "error_Ec": np.nan,
                    "error_Ep": np.nan,
                    "error_std_x": np.nan,
                    "error_std_v": np.nan,
                    "maximum_error": np.nan,
                    "correct": 0
                })

                continue


            Ec = np.mean([
                row["Ec"]
                for row in selected
            ])

            Ep = np.mean([
                row["Ep"]
                for row in selected
            ])

            mean_x = np.mean([
                row["mean_x"]
                for row in selected
            ])

            std_x = np.mean([
                row["std_x"]
                for row in selected
            ])

            mean_v = np.mean([
                row["mean_v"]
                for row in selected
            ])

            std_v = np.mean([
                row["std_v"]
                for row in selected
            ])


            error_Ec = (
                abs(Ec - 0.5)
                / 0.5
                * 100.0
            )

            error_Ep = (
                abs(Ep - 0.5)
                / 0.5
                * 100.0
            )

            error_std_x = (
                abs(std_x - 1.0)
                * 100.0
            )

            error_std_v = (
                abs(std_v - 1.0)
                * 100.0
            )


            maximum_error = max(
                error_Ec,
                error_Ep,
                error_std_x,
                error_std_v
            )


            correct = (
                valid_runs == len(seeds)
                and maximum_error
                < 100.0 * theory_tolerance
                and abs(mean_x)
                < mean_tolerance
                and abs(mean_v)
                < mean_tolerance
            )


            summary.append({
                "algorithm": algorithm,
                "eta": eta,
                "h": h,
                "valid_runs": valid_runs,
                "Ec": Ec,
                "Ep": Ep,
                "mean_x": mean_x,
                "std_x": std_x,
                "mean_v": mean_v,
                "std_v": std_v,
                "error_Ec": error_Ec,
                "error_Ep": error_Ep,
                "error_std_x": error_std_x,
                "error_std_v": error_std_v,
                "maximum_error": maximum_error,
                "correct": int(correct)
            })


with summary_file.open(
    "w",
    newline=""
) as file:

    names = [
        "algorithm",
        "eta",
        "h",
        "valid_runs",
        "Ec",
        "Ep",
        "mean_x",
        "std_x",
        "mean_v",
        "std_v",
        "error_Ec",
        "error_Ep",
        "error_std_x",
        "error_std_v",
        "maximum_error",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=names
    )

    writer.writeheader()
    writer.writerows(summary)


print()
print("===================================")
print("RESULTADOS")
print("===================================")
print()


for eta in eta_values:

    print(f"eta = {eta}")


    for h in h_values:

        print(f"  h = {h:g}")


        for algorithm in algorithms:

            selected = [
                row
                for row in summary
                if (
                    row["algorithm"]
                    == algorithm
                    and row["eta"] == eta
                    and row["h"] == h
                )
            ][0]


            if selected["valid_runs"] == 0:

                print(
                    f"    "
                    f"{algorithm_labels[algorithm]:17s} "
                    "INESTABLE"
                )

                continue


            if selected["correct"] == 1:
                state = "OK"
            else:
                state = "NO"


            print(
                f"    "
                f"{algorithm_labels[algorithm]:17s} "
                f"error máximo = "
                f"{selected['maximum_error']:7.2f} %   "
                f"{state}"
            )


    print()


reliable_h = {}


for algorithm in algorithms:

    valid_h = []


    for h in h_values:

        all_eta_correct = True


        for eta in eta_values:

            selected = [
                row
                for row in summary
                if (
                    row["algorithm"]
                    == algorithm
                    and row["eta"] == eta
                    and row["h"] == h
                )
            ][0]


            if selected["correct"] != 1:

                all_eta_correct = False


        if all_eta_correct:

            valid_h.append(h)


    if len(valid_h) > 0:

        reliable_h[algorithm] = max(
            valid_h
        )

    else:

        reliable_h[algorithm] = None


print()
print("===================================")
print("PASO MÁXIMO FIABLE")
print("===================================")


for algorithm in algorithms:

    h_max = reliable_h[algorithm]


    if h_max is None:

        print(
            f"{algorithm_labels[algorithm]}: "
            "ningún h cumple el criterio"
        )

    else:

        print(
            f"{algorithm_labels[algorithm]}: "
            f"h = {h_max:g}"
        )


candidates = [
    algorithm
    for algorithm in algorithms
    if reliable_h[algorithm]
    is not None
]


best_algorithm = None
best_h = None


if len(candidates) > 0:

    best_h = max([
        reliable_h[algorithm]
        for algorithm in candidates
    ])


    candidates = [
        algorithm
        for algorithm in candidates
        if reliable_h[algorithm]
        == best_h
    ]


    best_error = None


    for algorithm in candidates:

        errors = []


        for eta in eta_values:

            selected = [
                row
                for row in summary
                if (
                    row["algorithm"]
                    == algorithm
                    and row["eta"] == eta
                    and row["h"] == best_h
                )
            ][0]


            errors.append(
                selected["maximum_error"]
            )


        mean_error = np.mean(
            errors
        )


        if (
            best_error is None
            or mean_error < best_error
        ):

            best_error = mean_error
            best_algorithm = algorithm


print()


if best_algorithm is None:

    print(
        "No se ha podido seleccionar "
        "un algoritmo fiable."
    )

else:

    print(
        "Algoritmo seleccionado: "
        f"{algorithm_labels[best_algorithm]}"
    )

    print(
        "Paso temporal seleccionado: "
        f"h = {best_h:g}"
    )


for eta in eta_values:

    plt.figure(figsize=(8, 5))


    for algorithm in algorithms:

        errors = []


        for h in h_values:

            selected = [
                row
                for row in summary
                if (
                    row["algorithm"]
                    == algorithm
                    and row["eta"] == eta
                    and row["h"] == h
                )
            ][0]


            error = (
                selected["maximum_error"]
            )


            if np.isnan(error):

                errors.append(np.nan)

            else:

                errors.append(
                    max(error, 0.01)
                )


        plt.plot(
            h_values,
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
    plt.yscale("log")

    plt.xlabel(
        "Paso temporal $h$"
    )

    plt.ylabel(
        "Error máximo (%)"
    )

    plt.title(
        "Comparación de algoritmos "
        f"($\\eta={eta}$)"
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
        / f"harmonic_algorithms_eta_{eta_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


plt.figure(figsize=(8, 5))


for algorithm in algorithms:

    global_errors = []


    for h in h_values:

        errors = []


        for eta in eta_values:

            selected = [
                row
                for row in summary
                if (
                    row["algorithm"]
                    == algorithm
                    and row["eta"] == eta
                    and row["h"] == h
                )
            ][0]


            errors.append(
                selected["maximum_error"]
            )


        if np.all(
            np.isnan(errors)
        ):

            global_errors.append(
                np.nan
            )

        else:

            error = np.nanmax(
                errors
            )

            global_errors.append(
                max(error, 0.01)
            )


    plt.plot(
        h_values,
        global_errors,
        marker="o",
        label=algorithm_labels[algorithm]
    )


plt.axhline(
    100.0 * theory_tolerance,
    linestyle="--",
    label="Tolerancia"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(
    "Paso temporal $h$"
)

plt.ylabel(
    "Mayor error entre todos "
    "los valores de $\\eta$ (%)"
)

plt.title(
    "Comparación global "
    "de los algoritmos"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    figures_folder
    / "harmonic_algorithms_global.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print(
    "Resultados individuales guardados en:"
)
print(results_file)

print()
print("Resumen guardado en:")
print(summary_file)

print()
print("Gráficas guardadas en:")
print(figures_folder)