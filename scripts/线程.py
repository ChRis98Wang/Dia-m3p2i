import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# Academic‑style bar charts (one figure per environment)
# Fill / modify your data below as needed
# ===============================================================
success_rates = {
    "RL-MoCap":  [0.807, 0.790, 0.776],
    "RL-CMA":    [0.997, 0.987, 1.000],
    "RL-CMA*":   [0.992, 0.993, 1.000],
    "RL-BFGS*":  [0.995, 0.987, 0.963],
    "DIA-M3P2i(N_diff=2)" :[0.96, 0.91, 0.89],
    "DIA-M3P2i(N_diff=3)" :[0.98, 0.95, 0.95],
    "M3P2i(Halton)"       :[0.97, 0.96, 0.88],
}
environments = [
    "FetchPickDynamic-LiftedObstacles",
    "FetchPickDynamic-LiftedObstaclesX2",
    "FetchPickDynamic-LiftedObstaclesMaze",
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

algos = list(success_rates.keys())
n_algos = len(algos)

for env_idx, env_name in enumerate(environments):
    y_vals = [success_rates[a][env_idx] for a in algos]
    x = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(x, y_vals, width=0.6)

    # Annotate each bar with value
    for rect, val in zip(bars, y_vals):
        ax.annotate(f"{val:.3f}",
                    xy=(rect.get_x() + rect.get_width()/2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Task Success Rate")
    ax.set_title(env_name, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=25, ha="right")
    ax.margins(x=0.01)

    # Save high‑res PDF and PNG
    safe_name = env_name.replace("FetchPickDynamic-", "")
    #fig.savefig(f"{safe_name}_success_rate.pdf")
    #fig.savefig(f"{safe_name}_success_rate.png", dpi=300)

    plt.show()

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# Frequencies (Hz). Replace None with your results or add more algorithms.
envs = ["LiftedObstacles", "LiftedObstaclesX2", "LiftedObstaclesMaze"]
freq = {
    "CMA-ES": [0.43, 0.65, 0.71],
    "CMA-ES-J": [4.57, 5.78, 3.63],
    "SEP-CMA-ES": [0.99, 0.51, 0.47],
    "SEP-CMA-ES-J": [3.11, 3.44, 1.75],
    "CMA-ES-wM": [1.10, 0.62, 0.72],
    "CMA-ES-wM-J": [4.56, 5.68, 4.13],
    "Gaussian-Sampling": [1.02, 3.51, 1.81],
    "Gaussian-Sampling-J": [3.89, 3.62, 4.59],
    "CG-G": [2.01, 5.56, 4.85],
    "Powell-G": [2.80, 3.66, 3.97],
    "Nelder-Mead-G": [0.43, 5.26, 4.90],
    "Trust-Constr-G": [0.21, 0.02, 0.02],
    "COBYLA-G": [1.34, 0.58, 0.65],
    "SLSQP": [0.90, 0.68, 0.76],
    "BFGS-G": [10.09, 6.20, 5.78],
    "L-BFGS-B-G": [8.88, 6.85, 5.99],
    # --- your algorithm ---
    "DIA-M3P2i(N_diff=2)":      [6.8, 6.8, 2.6],
"DIA-M3P2i(N_diff=3)":      [4.2, 4.2, 1.5],
"M3P2i(Halton)": [10, 10, 6],
}

plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for idx, ax in enumerate(axes):
    env = envs[idx]
    # filter None and sort descending
    items = [(alg, hz[idx]) for alg, hz in freq.items() if hz[idx] is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    x = np.arange(len(labels))

    ax.bar(x, values, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_title(env, fontsize=12, pad=10)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

axes[0].set_ylabel("step frequency [Hz]")
plt.tight_layout()
plt.show()
for idx, env in enumerate(envs):
    # filter and sort
    items = [(alg, hz[idx]) for alg, hz in freq.items() if hz[idx] is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, values, color="#1f77b4")
    plt.xticks(x, labels, rotation=65, ha="right", fontsize=8)
    plt.ylabel("step frequency [Hz]")
    plt.title(f"{env}", pad=10, fontsize=12)
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(f"{env}_frequency.png",dpi=300)
    plt.show()
