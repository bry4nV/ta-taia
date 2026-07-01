import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data import MinMaxScaler, load_adjacency, load_splits
from ..metrics import masked_mae_loss, metrics_by_horizon
from ..models import LSTTNVariant
from .common import save_json, set_seed
from .pretrain import load_pretrained_transformer


def build_forecasting_model(cfg, checkpoint_path, parameters, device):
    transformer, _ = load_pretrained_transformer(checkpoint_path, device)
    adjacency, forward_adj, backward_adj = load_adjacency(cfg.graph_path)
    model = LSTTNVariant(
        transformer=transformer,
        num_nodes=adjacency.shape[0],
        pred_len=cfg.pred_len,
        forward_adj=forward_adj,
        backward_adj=backward_adj,
        long_hidden=parameters["long_hidden"],
        period_hidden=parameters["period_hidden"],
        short_hidden=parameters["short_hidden"],
        fusion=parameters.get("fusion", "attention"),
        fusion_heads=parameters.get("fusion_heads", 4),
        dropout=parameters["dropout"],
    ).to(device)
    model.freeze_transformer()
    return model


def _forecast_loss(model, loader, device, null_value, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for x_short, x_long, x_day, x_week, target in loader:
        tensors = [x.to(device, non_blocking=True) for x in (x_short, x_long, x_day, x_week, target)]
        x_short, x_long, x_day, x_week, target = tensors
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                prediction = model(x_short, x_long, x_day, x_week).transpose(1, 2)
                loss = masked_mae_loss(prediction, target, null_value)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), 3.0
                )
                scaler.step(optimizer)
                scaler.update()
        total += loss.item() * target.size(0)
        count += target.size(0)
    return total / max(count, 1)


def make_forecast_loaders(cfg, device, batch_size=None):
    _, datasets, _ = load_splits(cfg.dataset_dir, cfg.short_len, cfg.long_len, cfg.pred_len)
    batch_size = batch_size or cfg.forecast_batch_size
    common = dict(batch_size=batch_size, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def train_forecasting(
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    parameters: dict,
    device: torch.device,
    run_name: str,
    max_epochs=None,
    trial=None,
    evaluate_test: bool = True,
):
    set_seed(cfg.seed)
    model = build_forecasting_model(cfg, checkpoint_path, parameters, device)
    loaders = make_forecast_loaders(cfg, device, parameters.get("batch_size"))
    scaler = MinMaxScaler.from_dataset(cfg.dataset_dir)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=parameters["learning_rate"],
        weight_decay=parameters.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    max_epochs = max_epochs or cfg.forecast_epochs
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss = _forecast_loss(
            model, loaders["train"], device, scaler.normalized_zero, optimizer, amp_scaler
        )
        valid_loss = _forecast_loss(model, loaders["valid"], device, scaler.normalized_zero)
        scheduler.step(valid_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"[{run_name}] {epoch:03d} train={train_loss:.6f} val={valid_loss:.6f}")

        if trial is not None:
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise RuntimeError("OPTUNA_PRUNED")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.forecast_patience:
                break

    model.load_state_dict(best_state)
    run_dir = cfg.output_dir / "forecasting" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Optuna y las pruebas cortas nunca consultan test.
    test_metrics = evaluate_model(model, loaders["test"], scaler, device) if evaluate_test else None
    if trial is None:
        torch.save({"model_state": best_state, "parameters": parameters}, run_dir / "best.pt")
    summary = {
        "best_valid_loss_normalized": best_loss,
        "test": test_metrics,
        "seconds": time.time() - start,
        "parameters": parameters,
        "history": history,
    }
    save_json(summary, run_dir / "metrics.json")
    return summary


@torch.no_grad()
def evaluate_model(model, loader, scaler, device):
    model.eval()
    predictions, targets = [], []
    for x_short, x_long, x_day, x_week, target in loader:
        x_short, x_long, x_day, x_week = [
            tensor.to(device, non_blocking=True) for tensor in (x_short, x_long, x_day, x_week)
        ]
        prediction = model(x_short, x_long, x_day, x_week).transpose(1, 2).cpu().numpy()
        predictions.append(prediction)
        targets.append(target.numpy())
    prediction = scaler.inverse(np.concatenate(predictions))
    target = scaler.inverse(np.concatenate(targets))
    return metrics_by_horizon(target, prediction)
