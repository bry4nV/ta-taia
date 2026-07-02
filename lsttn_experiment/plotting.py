"""Gráficas reproducibles para todas las fases del experimento."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(history, path, title, ylabel="Pérdida"):
    if not history:
        return
    path = _prepare_path(path)
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    valid = [row["valid_loss"] for row in history]
    has_train_eval = all("train_eval_loss" in row for row in history)
    best_index = int(np.argmin(valid))

    if has_train_eval:
        train_eval = [row["train_eval_loss"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].plot(epochs, train, label="Entrenamiento online", linewidth=1.8)
        axes[0].set(title="Optimización", xlabel="Época", ylabel=ylabel)
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(epochs, train_eval, label="Train (eval)", linewidth=1.8)
        axes[1].plot(epochs, valid, label="Validación (eval)", linewidth=1.8)
        axes[1].scatter(
            epochs[best_index], valid[best_index], color="crimson", zorder=3,
            label=f"Mejor validación: época {epochs[best_index]}",
        )
        axes[1].set(title="Generalización", xlabel="Época", ylabel=ylabel)
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.suptitle(title)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Train es el promedio online mientras se actualizan los pesos (model.train());
        # valid se calcula después de la época con pesos fijos (model.eval()).
        ax.plot(epochs, train, label="Entrenamiento online", linewidth=1.8)
        ax.plot(epochs, valid, label="Validación (eval)", linewidth=1.8)
        ax.scatter(epochs[best_index], valid[best_index], color="crimson", zorder=3,
                   label=f"Mejor validación: época {epochs[best_index]}")
        ax.set(title=title, xlabel="Época", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_candidate_comparison(rows, path):
    """Compara reconstrucción y, cuando existan, probes de forecasting."""
    if not rows:
        return
    path = _prepare_path(path)
    names = [row["name"] for row in rows]
    reconstruction = [row["reconstruction"] for row in rows]
    probes = [row.get("probe") for row in rows]
    has_probes = any(value is not None for value in probes)

    fig, axes = plt.subplots(1, 2 if has_probes else 1, figsize=(10 if has_probes else 6, 4.5))
    axes = np.atleast_1d(axes)
    axes[0].bar(names, reconstruction, color="#4C78A8")
    axes[0].set(title="Reconstrucción del MST", ylabel="MAE normalizado")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(reconstruction):
        axes[0].text(index, value, f"{value:.4f}", ha="center", va="bottom")

    if has_probes:
        values = [np.nan if value is None else value for value in probes]
        axes[1].bar(names, values, color="#F58518")
        axes[1].set(title="Probe de forecasting", ylabel="MAE normalizado")
        axes[1].grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            if np.isfinite(value):
                axes[1].text(index, value, f"{value:.4f}", ha="center", va="bottom")
    fig.suptitle("Comparación de candidatos")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_horizon_metrics(metrics, path):
    by_horizon = metrics.get("by_horizon", {})
    if not by_horizon:
        return
    path = _prepare_path(path)
    labels = list(by_horizon)
    minutes = [int(label.removesuffix("min")) for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, ("MAE", "RMSE", "MAPE")):
        values = [by_horizon[label][metric] for label in labels]
        ax.plot(minutes, values, marker="o", linewidth=1.8)
        ax.set(title=metric, xlabel="Horizonte (minutos)", ylabel=metric)
        ax.set_xticks(minutes)
        ax.grid(alpha=0.25)
    fig.suptitle("Métricas de test por horizonte")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prediction_example(target, prediction, path, selection="representative"):
    if target.size == 0:
        return
    path = _prepare_path(path)
    if selection == "high_variability":
        # Caso de estrés reproducible: sensor más variable del primer pronóstico.
        sample = 0
        sensor = int(np.argmax(np.ptp(target[sample], axis=0)))
        description = "Caso de alta variabilidad"
    elif selection == "representative":
        # Serie cuyo MAE se aproxima a la mediana del conjunto de test. Se ignoran
        # ceros igual que en las métricas enmascaradas del experimento.
        mask = ~np.isclose(target, 0.0)
        error_sum = np.where(mask, np.abs(target - prediction), 0.0).sum(axis=1)
        counts = mask.sum(axis=1)
        series_mae = np.divide(
            error_sum,
            counts,
            out=np.full_like(error_sum, np.nan, dtype=float),
            where=counts > 0,
        )
        finite = np.isfinite(series_mae)
        median_mae = float(np.median(series_mae[finite]))
        distance = np.where(finite, np.abs(series_mae - median_mae), np.inf)
        sample, sensor = np.unravel_index(np.argmin(distance), distance.shape)
        description = "Caso representativo (MAE mediano)"
    else:
        raise ValueError(f"Selección de ejemplo desconocida: {selection}")

    minutes = np.arange(1, target.shape[1] + 1) * 5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(minutes, target[sample, :, sensor], marker="o", label="Real")
    ax.plot(minutes, prediction[sample, :, sensor], marker="o", label="Pronóstico")
    ax.set(title=f"{description} — sensor {sensor}", xlabel="Horizonte (minutos)",
           ylabel="Flujo de tráfico")
    ax.set_xticks(minutes)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_optuna_study(study, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = [
        trial for trial in study.trials
        if trial.value is not None and trial.state.name == "COMPLETE"
    ]
    pruned = [
        trial for trial in study.trials
        if trial.value is not None and trial.state.name == "PRUNED"
    ]
    if not completed:
        return

    numbers = [trial.number for trial in completed]
    values = [trial.value for trial in completed]
    running_best = np.minimum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(numbers, values, alpha=0.7, label="Completado")
    if pruned:
        ax.scatter(
            [trial.number for trial in pruned],
            [trial.value for trial in pruned],
            alpha=0.55,
            marker="x",
            color="gray",
            label="Podado",
        )
    ax.plot(numbers, running_best, color="crimson", label="Mejor acumulado")
    ax.set(title="Optimización de hiperparámetros", xlabel="Trial", ylabel="MAE de validación")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optuna_history.png", dpi=180)
    plt.close(fig)

    try:
        import optuna

        importance = optuna.importance.get_param_importances(study)
    except (ImportError, ValueError, RuntimeError):
        importance = {}
    if importance:
        names = list(importance)[::-1]
        values = [importance[name] for name in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, values, color="#54A24B")
        ax.set(title="Importancia de hiperparámetros", xlabel="Importancia")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "optuna_importance.png", dpi=180)
        plt.close(fig)
