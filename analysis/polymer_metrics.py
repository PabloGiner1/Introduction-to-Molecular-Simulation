import numpy as np


def compute_time_averaged_msd(trajectory):

    if trajectory.ndim != 2 or trajectory.shape[1] != 3:

        raise ValueError(
            "La trayectoria debe tener forma (n_muestras, 3)."
        )

    if len(trajectory) < 8:

        raise ValueError(
            "Se necesitan al menos 8 muestras para calcular el MSD."
        )

    max_lag = len(trajectory) // 4

    lag_indices = np.unique(
        np.linspace(
            1,
            max_lag,
            min(100, max_lag),
            dtype=int
        )
    )

    msd = np.array([
        np.mean(
            np.sum(
                (
                    trajectory[lag:]
                    - trajectory[:-lag]
                ) ** 2,
                axis=1
            )
        )
        for lag in lag_indices
    ])

    return lag_indices, msd
