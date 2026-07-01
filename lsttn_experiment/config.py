from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TransformerCandidate:
    name: str
    d_model: int
    num_layers: int = 4
    num_heads: int = 4

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model debe ser divisible entre num_heads")


@dataclass
class ExperimentConfig:
    """Configuración compartida para que los candidatos sean comparables."""

    project_root: Path = PROJECT_ROOT
    dataset_name: str = "PEMS08"
    short_len: int = 12
    long_len: int = 2016  # una semana a intervalos de cinco minutos
    pred_len: int = 12
    patch_size: int = 12
    mask_ratio: float = 0.75
    dropout: float = 0.1
    seed: int = 42

    pretrain_epochs: int = 100
    pretrain_patience: int = 10
    pretrain_batch_size: int = 8
    pretrain_lr: float = 1e-3

    forecast_epochs: int = 100
    forecast_patience: int = 10
    forecast_batch_size: int = 16
    num_workers: int = 4

    candidates: Tuple[TransformerCandidate, ...] = field(
        default_factory=lambda: (
            TransformerCandidate("mst_d64", d_model=64),
            TransformerCandidate("mst_d96", d_model=96),
        )
    )

    @property
    def dataset_dir(self) -> Path:
        return self.project_root / "dataset" / self.dataset_name

    @property
    def graph_path(self) -> Path:
        return self.project_root / "dataset" / "sensor_graph" / "adj_mx_08.pkl"

    @property
    def output_dir(self) -> Path:
        path = self.project_root / "resultados_modular"
        path.mkdir(parents=True, exist_ok=True)
        return path
