from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
parameters_file = project_root / "data" / "raw" / "parameters.txt"

parameters = {}
with parameters_file.open() as file:
    for line in file:
        name, value = line.strip().split("=", maxsplit=1)
        parameters[name] = float(value)

eta = parameters["eta"]
h = parameters["h"]

# 1. Cargar los datos guardados desde C (datos_eta_h.txt)
# Columnas: 0: t, 1: x, 2: p, 3: E_c, 4: E_p
data_file = project_root / "data" / "raw" / f"datos_{eta:.1f}_{h:.4f}.txt"
data = np.loadtxt(data_file)
carpeta = project_root / "data" / "raw"


t = data[:, 0]
x = data[:, 1]
p = data[:, 2]
E_c = data[:, 3]
E_p = data[:, 4]

# 2. Descartar el periodo de termalización (primer 20% del tiempo)
t_term = 0.2 * t[-1]
mask = t >= t_term

x_eq = x[mask]
p_eq = p[mask]
E_c_eq = E_c[mask]
E_p_eq = E_p[mask]

# 3. Calcular promedios y desviaciones estándar numéricas
mean_Ec = np.mean(E_c_eq)
mean_Ep = np.mean(E_p_eq)
mean_Etot = np.mean(E_c_eq + E_p_eq)

mean_x, std_x = np.mean(x_eq), np.std(x_eq)
mean_p, std_p = np.mean(p_eq), np.std(p_eq)


# =======================================================
# FIGURA 1: ENERGÍAS INSTANTÁNEAS Y PROMEDIOS
# =======================================================
plt.figure(figsize=(8, 5))
plt.plot(t, E_c, label="$E_{cin}$", alpha=0.6, color='tab:blue')
plt.plot(t, E_p, label="$E_{pot}$", alpha=0.6, color='tab:orange')
plt.axhline(0.5, color='black', linestyle='--', label="Equipartición ($0.5 k_B T$)")
plt.xlabel("Tiempo $t$")
plt.ylabel("Energía")
plt.title("Evolución Temporal de las Energías Instantáneas")
plt.legend(loc="upper right")
plt.grid(True)

text_fig1 = (f"Promedios ($t \\geq {t_term:.1f}$):\n"
             f"$\\langle E_{{cin}} \\rangle$ = {mean_Ec:.4f}  (Teórico: 0.5000)\n"
             f"$\\langle E_{{pot}} \\rangle$ = {mean_Ep:.4f}  (Teórico: 0.5000)\n"
             f"$\\langle E_{{tot}} \\rangle$ = {mean_Etot:.4f}  (Teórico: 1.0000)")
plt.gca().text(0.03, 0.95, text_fig1, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig("energias_instantaneas_0,001_0,1.png", dpi=300, bbox_inches='tight')


# =======================================================
# FIGURA 2: CONVERGENCIA DE LOS PROMEDIOS ACUMULADOS
# =======================================================
plt.figure(figsize=(8, 5))
cum_Ec = np.cumsum(E_c) / (np.arange(len(E_c)) + 1)
cum_Ep = np.cumsum(E_p) / (np.arange(len(E_p)) + 1)

plt.plot(t, cum_Ec, label=r"$\langle E_{cin} \rangle_{acum}$", color='tab:blue')
plt.plot(t, cum_Ep, label=r"$\langle E_{pot} \rangle_{acum}$", color='tab:orange')
plt.axhline(0.5, color='black', linestyle='--', label="Valor Teórico ($0.5$)")
plt.xlabel("Tiempo $t$")
plt.ylabel("Energía Media Acumulada")
plt.title("Convergencia del Promedio Temporal de Energía")
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.savefig("energias_convergencia_0,001_0,1.png", dpi=300, bbox_inches='tight')


# =======================================================
# FIGURA 3: DISTRIBUCIÓN DE POSICIONES P(x)
# =======================================================
plt.figure(figsize=(8, 5))
plt.hist(x_eq, bins=60, density=True, alpha=0.6, color='royalblue', label="Simulación")
x_grid = np.linspace(min(x_eq), max(x_eq), 200)
P_x_teorico = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-x_grid**2 / 2.0)
plt.plot(x_grid, P_x_teorico, 'r-', lw=2, label="Gaussiana Teórica $\\mathcal{N}(0,1)$")
plt.xlabel("Posición $x$")
plt.ylabel("Densidad de probabilidad $P(x)$")
plt.title("Distribución de Posiciones $P(x)$")
plt.legend(loc="upper right")
plt.grid(True)

text_fig3 = (f"Estadística de $x$ ($t \\geq {t_term:.1f}$):\n"
             f"$\\mu_x$ = {mean_x:.4f}  (Teórico: 0.0000)\n"
             f"$\\sigma_x$ = {std_x:.4f}  (Teórico: 1.0000)")
plt.gca().text(0.03, 0.95, text_fig3, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig("distribucion_posiciones_0,001_0,1.png", dpi=300, bbox_inches='tight')


# =======================================================
# FIGURA 4: DISTRIBUCIÓN DE VELOCIDADES P(p)
# =======================================================
plt.figure(figsize=(8, 5))
plt.hist(p_eq, bins=60, density=True, alpha=0.6, color='mediumseagreen', label="Simulación")
p_grid = np.linspace(min(p_eq), max(p_eq), 200)
P_p_teorico = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-p_grid**2 / 2.0)
plt.plot(p_grid, P_p_teorico, 'r-', lw=2, label="Gaussiana Teórica $\\mathcal{N}(0,1)$")
plt.xlabel("Momento / Velocidad $p$")
plt.ylabel("Densidad de probabilidad $P(p)$")
plt.title("Distribución de Velocidades $P(p)$")
plt.legend(loc="upper right")
plt.grid(True)

text_fig4 = (f"Estadística de $p$ ($t \\geq {t_term:.1f}$):\n"
             f"$\\mu_p$ = {mean_p:.4f}  (Teórico: 0.0000)\n"
             f"$\\sigma_p$ = {std_p:.4f}  (Teórico: 1.0000)")
plt.gca().text(0.03, 0.95, text_fig4, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
plt.savefig("distribucion_velocidades_0,001_0,1.png", dpi=300, bbox_inches='tight')

print("¡Las 4 imágenes se han guardado correctamente en la carpeta del proyecto!")
plt.show()