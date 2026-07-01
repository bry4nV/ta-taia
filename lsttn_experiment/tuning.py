from pathlib import Path

import optuna

from .config import ExperimentConfig
from .training.forecast import train_forecasting


def tune_forecasting(
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    device,
    n_trials: int,
    storage: str,
    study_name: str,
):
    def objective(trial):
        d_model = int(__import__("torch").load(checkpoint_path, map_location="cpu")["candidate"]["d_model"])
        possible_heads = [heads for heads in (2, 4, 8) if d_model % heads == 0]
        parameters = {
            "long_hidden": trial.suggest_categorical("long_hidden", [4, 8, 16, 32]),
            "period_hidden": trial.suggest_categorical("period_hidden", [4, 8, 16, 32]),
            "short_hidden": trial.suggest_categorical("short_hidden", [32, 64, 96, 128]),
            "fusion": "attention",
            "fusion_heads": trial.suggest_categorical("fusion_heads", possible_heads),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }
        try:
            summary = train_forecasting(
                cfg, checkpoint_path, parameters, device,
                run_name=f"trial_{trial.number}", max_epochs=20, trial=trial, evaluate_test=False,
            )
        except RuntimeError as error:
            if str(error) == "OPTUNA_PRUNED":
                raise optuna.TrialPruned() from error
            raise
        return summary["best_valid_loss_normalized"]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
