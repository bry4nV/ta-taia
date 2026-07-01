import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, TransformerCandidate
from ..data import load_splits
from ..models import MaskedSubseriesTransformer
from ..plotting import plot_training_history
from .common import save_json, set_seed


def _mean_reconstruction_loss(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.L1Loss()
    total = 0.0
    count = 0
    for windows in loader:
        windows = windows.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                reconstruction, labels = model(windows, mode="pretrain")
                loss = criterion(reconstruction, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        total += loss.item() * windows.size(0)
        count += windows.size(0)
    return total / max(count, 1)


def pretrain_candidate(cfg: ExperimentConfig, candidate: TransformerCandidate, device: torch.device):
    set_seed(cfg.seed)
    _, _, pretrain_sets = load_splits(
        cfg.dataset_dir, cfg.short_len, cfg.long_len, cfg.pred_len
    )
    train_loader = DataLoader(
        pretrain_sets["train"], batch_size=cfg.pretrain_batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        pretrain_sets["valid"], batch_size=cfg.pretrain_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    model = MaskedSubseriesTransformer(
        patch_size=cfg.patch_size,
        d_model=candidate.d_model,
        num_heads=candidate.num_heads,
        num_layers=candidate.num_layers,
        long_len=cfg.long_len,
        mask_ratio=cfg.mask_ratio,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = cfg.output_dir / "pretraining" / candidate.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    patience = 0
    history = []
    start = time.time()
    for epoch in range(1, cfg.pretrain_epochs + 1):
        train_loss = _mean_reconstruction_loss(model, train_loader, device, optimizer, scaler)
        # Misma secuencia de máscaras de validación para comparar candidatos.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(cfg.seed + 10_000)
            valid_loss = _mean_reconstruction_loss(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"[{candidate.name}] {epoch:03d} train={train_loss:.6f} val={valid_loss:.6f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "candidate": asdict(candidate),
                "long_len": cfg.long_len,
                "patch_size": cfg.patch_size,
                "mask_ratio": cfg.mask_ratio,
                "dropout": cfg.dropout,
                "best_valid_loss": best_loss,
            }, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.pretrain_patience:
                break

    summary = {
        "candidate": asdict(candidate),
        "best_valid_loss": best_loss,
        "epochs_completed": len(history),
        "seconds": time.time() - start,
        "history": history,
    }
    save_json(summary, run_dir / "metrics.json")
    plot_training_history(
        history,
        run_dir / "learning_curve.png",
        f"Preentrenamiento {candidate.name}",
        ylabel="MAE de reconstrucción normalizado",
    )
    return summary


def load_pretrained_transformer(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    candidate = checkpoint["candidate"]
    model = MaskedSubseriesTransformer(
        patch_size=checkpoint["patch_size"],
        d_model=candidate["d_model"],
        num_heads=candidate["num_heads"],
        num_layers=candidate["num_layers"],
        long_len=checkpoint["long_len"],
        mask_ratio=checkpoint["mask_ratio"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
