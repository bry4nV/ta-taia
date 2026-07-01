import numpy as np
import torch


def masked_mae_loss(prediction, target, null_value: float):
    mask = ~torch.isclose(target, torch.as_tensor(null_value, device=target.device, dtype=target.dtype))
    mask = mask.float()
    mask = mask / mask.mean().clamp_min(1e-8)
    loss = torch.abs(prediction - target) * mask
    return torch.nan_to_num(loss).mean()


def regression_metrics(target, prediction, null_value: float = 0.0):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    mask = ~np.isclose(target, null_value)
    error = prediction[mask] - target[mask]
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    mape = np.mean(np.abs(error) / np.maximum(np.abs(target[mask]), 1e-8)) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def metrics_by_horizon(target, prediction):
    by_horizon = {
        f"{(index + 1) * 5}min": regression_metrics(target[:, index], prediction[:, index])
        for index in range(target.shape[1])
    }
    result = {
        "global": regression_metrics(target, prediction),
        "by_horizon": by_horizon,
    }
    # Alias directos para facilitar la comparación con la tabla del paper.
    for name in ("15min", "30min", "60min"):
        if name in by_horizon:
            result[name] = by_horizon[name]
    return result
