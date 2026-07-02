#!/usr/bin/env python3
"""CLI del experimento modular LSTTN-PEMS08."""

import argparse
import json
from pathlib import Path

from lsttn_experiment.config import ExperimentConfig
from lsttn_experiment.plotting import (
    plot_candidate_comparison,
    plot_optuna_study,
    plot_training_history,
)
from lsttn_experiment.training.common import resolve_device, save_json
from lsttn_experiment.training.forecast import train_forecasting
from lsttn_experiment.training.pretrain import pretrain_candidate


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="Preentrena un candidato MST")
    pretrain.add_argument("--candidate", choices=("mst_d64", "mst_d96"), required=True)
    pretrain.add_argument("--device", default="cuda:0")

    compare = subparsers.add_parser("compare", help="Compara pérdidas de validación")

    probe = subparsers.add_parser("probe", help="Forecasting corto sin consultar test")
    probe.add_argument("--candidate", choices=("mst_d64", "mst_d96"), required=True)
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--epochs", type=int, default=20)

    tune = subparsers.add_parser("tune", help="Optuna con Transformer congelado")
    tune.add_argument("--checkpoint", type=Path, required=True)
    tune.add_argument("--device", default="cuda:0")
    tune.add_argument("--trials", type=int, default=20)
    tune.add_argument("--storage", default="sqlite:///resultados_modular/optuna.db")
    tune.add_argument("--study-name", default="lsttn_pems08")
    tune.add_argument(
        "--sampler-seed", type=int, default=None,
        help="Semilla del muestreador Optuna; use una distinta por worker paralelo",
    )

    train = subparsers.add_parser("train", help="Entrena y evalúa la configuración final")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--params", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--run-name", default="final")
    return parser.parse_args()


def compare_candidates(cfg):
    results = []
    plot_rows = []
    for candidate in cfg.candidates:
        path = cfg.output_dir / "pretraining" / candidate.name / "metrics.json"
        if path.exists():
            with path.open(encoding="utf-8") as file:
                metrics = json.load(file)
            probe_path = cfg.output_dir / "forecasting" / f"probe_{candidate.name}" / "metrics.json"
            probe_loss = None
            if probe_path.exists():
                with probe_path.open(encoding="utf-8") as file:
                    probe_metrics = json.load(file)
                probe_loss = probe_metrics["best_valid_loss_normalized"]
                plot_training_history(
                    probe_metrics.get("history", []),
                    probe_path.parent / "learning_curve.png",
                    f"Probe {candidate.name}",
                    ylabel="MAE normalizado",
                )
            results.append((candidate.name, metrics["best_valid_loss"], probe_loss, metrics["seconds"]))
            plot_training_history(
                metrics.get("history", []),
                path.parent / "learning_curve.png",
                f"Preentrenamiento {candidate.name}",
                ylabel="MAE de reconstrucción normalizado",
            )
            plot_rows.append({
                "name": candidate.name,
                "reconstruction": metrics["best_valid_loss"],
                "probe": probe_loss,
            })
    if not results:
        raise FileNotFoundError("No hay candidatos preentrenados para comparar")
    for name, loss, probe_loss, seconds in sorted(results, key=lambda row: row[1]):
        probe_text = "pendiente" if probe_loss is None else f"{probe_loss:.6f}"
        print(
            f"{name:10s} reconstrucción={loss:.6f} "
            f"forecast_val={probe_text} tiempo={seconds / 3600:.2f}h"
        )
    completed_probes = [row for row in results if row[2] is not None]
    if completed_probes:
        print(f"Ganador por forecasting de validación: {min(completed_probes, key=lambda row: row[2])[0]}")
    else:
        print(f"Ganador provisional por reconstrucción: {min(results, key=lambda row: row[1])[0]}")
    plot_candidate_comparison(plot_rows, cfg.output_dir / "candidate_comparison.png")
    print(f"Gráficas guardadas en: {cfg.output_dir}")


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    if args.command == "pretrain":
        candidate = next(item for item in cfg.candidates if item.name == args.candidate)
        pretrain_candidate(cfg, candidate, resolve_device(args.device))
    elif args.command == "compare":
        compare_candidates(cfg)
    elif args.command == "probe":
        checkpoint = cfg.output_dir / "pretraining" / args.candidate / "best.pt"
        parameters = {
            "long_hidden": 16,
            "period_hidden": 16,
            "short_hidden": 64,
            "fusion": "attention",
            "fusion_heads": 4,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": cfg.forecast_batch_size,
        }
        train_forecasting(
            cfg, checkpoint, parameters, resolve_device(args.device),
            run_name=f"probe_{args.candidate}", max_epochs=args.epochs, evaluate_test=False,
        )
    elif args.command == "tune":
        from lsttn_experiment.tuning import tune_forecasting

        study = tune_forecasting(
            cfg, args.checkpoint, resolve_device(args.device), args.trials,
            args.storage, args.study_name, args.sampler_seed,
        )
        parameters = {
            **study.best_params,
            "fusion": "attention",
        }
        save_json(
            {"best_validation_loss": study.best_value, "parameters": parameters},
            cfg.output_dir / "optuna_best.json",
        )
        plot_optuna_study(study, cfg.output_dir / "optuna_plots")
        study.trials_dataframe().to_csv(cfg.output_dir / "optuna_trials.csv", index=False)
        print("Mejor valor:", study.best_value)
        print("Parámetros:", parameters)
    elif args.command == "train":
        with args.params.open(encoding="utf-8") as file:
            content = json.load(file)
        parameters = content.get("parameters", content)
        train_forecasting(
            cfg, args.checkpoint, parameters, resolve_device(args.device), args.run_name
        )


if __name__ == "__main__":
    main()
