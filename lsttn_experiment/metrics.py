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
    horizons = {"15min": 2, "30min": 5, "60min": 11}
    result = {"global": regression_metrics(target, prediction)}
    for name, index in horizons.items():
        result[name] = regression_metrics(target[:, index], prediction[:, index])
    return result
