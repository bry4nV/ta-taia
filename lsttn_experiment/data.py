import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


def load_pickle(path: Path):
    with path.open("rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


class MinMaxScaler:
    """Escalador del dataset oficial: valores normalizados en [-1, 1]."""

    def __init__(self, minimum: float, maximum: float):
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.scale = self.maximum - self.minimum

    @classmethod
    def from_dataset(cls, dataset_dir: Path):
        minimum = np.asarray(load_pickle(dataset_dir / "min.pkl")).reshape(-1)[0]
        maximum = np.asarray(load_pickle(dataset_dir / "max.pkl")).reshape(-1)[0]
        return cls(minimum, maximum)

    def inverse(self, values):
        return (values + 1.0) * 0.5 * self.scale + self.minimum

    @property
    def normalized_zero(self) -> float:
        return 2.0 * (0.0 - self.minimum) / self.scale - 1.0


def _valid_indices(indices: Iterable[Tuple[int, int, int]], required_history: int):
    """Conserva los splits oficiales y descarta solo ventanas sin historia suficiente."""
    return [tuple(map(int, idx)) for idx in indices if int(idx[1]) >= required_history]


class ForecastDataset(Dataset):
    """Genera ventanas al vuelo, sin materializar decenas de GB en RAM."""

    def __init__(self, data: np.ndarray, indices, short_len: int, long_len: int, pred_len: int):
        self.data = data
        self.short_len = short_len
        self.long_len = long_len
        self.pred_len = pred_len
        # Se necesita una hora adicional para la referencia semanal.
        self.indices = _valid_indices(indices, long_len + short_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        _, current, _ = self.indices[item]
        flow = self.data[..., 0]
        x_short = self.data[current - self.short_len : current]
        x_long = flow[current - self.long_len : current]
        x_day = flow[current - 288 - self.short_len : current - 288]
        x_week = flow[current - self.long_len - self.short_len : current - self.long_len]
        target = flow[current : current + self.pred_len]
        return tuple(torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in (
            x_short, x_long, x_day, x_week, target
        ))


class PretrainDataset(Dataset):
    def __init__(self, data: np.ndarray, indices, long_len: int):
        self.flow = data[..., 0]
        self.long_len = long_len
        self.indices = _valid_indices(indices, long_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        current = self.indices[item][1]
        window = np.asarray(self.flow[current - self.long_len : current], dtype=np.float32)
        return torch.from_numpy(window)


def load_splits(dataset_dir: Path, short_len: int, long_len: int, pred_len: int):
    data = np.asarray(load_pickle(dataset_dir / "data.pkl"), dtype=np.float32)
    raw_indices: Dict[str, list] = {
        "train": load_pickle(dataset_dir / "train_index.pkl"),
        "valid": load_pickle(dataset_dir / "valid_index.pkl"),
        "test": load_pickle(dataset_dir / "test_index.pkl"),
    }
    data[~np.isfinite(data)] = np.nan
    train_end = int(raw_indices["train"][-1][1])
    means = np.nanmean(data[:train_end], axis=0, keepdims=True)
    data = np.where(np.isnan(data), means, data).astype(np.float32)
    forecast = {
        split: ForecastDataset(data, idx, short_len, long_len, pred_len)
        for split, idx in raw_indices.items()
    }
    pretrain = {
        split: PretrainDataset(data, idx, long_len)
        for split, idx in raw_indices.items()
        if split != "test"
    }
    return data, forecast, pretrain


def load_adjacency(path: Path):
    obj = load_pickle(path)
    adjacency = obj[2] if isinstance(obj, tuple) and len(obj) == 3 else obj
    if sparse.issparse(adjacency):
        adjacency = adjacency.toarray()
    adjacency = np.asarray(adjacency, dtype=np.float32)

    def transition(matrix):
        degree = matrix.sum(axis=1)
        inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree != 0)
        return (inverse[:, None] * matrix).astype(np.float32)

    forward = transition(adjacency).T
    backward = transition(adjacency.T).T
    return adjacency, torch.from_numpy(forward), torch.from_numpy(backward)
