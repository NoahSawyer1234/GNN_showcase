"""Small helpers shared by the dataset -> graph preprocessing scripts."""
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# ------------------------------- splitting --------------------------------
def stratified_split(y: Sequence[int], fractions=(0.6, 0.2, 0.2), seed: int = 0):
    """Class-balanced random split. Returns (train_idx, val_idx, test_idx) as sorted numpy arrays."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr = int(round(fractions[0] * len(idx)))
        n_va = int(round(fractions[1] * len(idx)))
        parts[0].append(idx[:n_tr])
        parts[1].append(idx[n_tr:n_tr + n_va])
        parts[2].append(idx[n_tr + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


def index_to_mask(idx, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.as_tensor(np.asarray(idx), dtype=torch.long)] = True
    return mask


# ----------------------------- standardisation -----------------------------
class Standardizer:
    """Per-column z-scoring. Always `fit` on training rows only to avoid leakage."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return cls(np.zeros(x.shape[1]), np.ones(x.shape[1]))
        mean, std = x.mean(0), x.std(0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ------------------------------- persistence -------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_processed(path, payload, meta: dict):
    """Save `payload` (a PyG Data / dict / list of Data) plus a JSON-able `meta` dict.

    Writes `<path>` (torch pickle) and `<path>.meta.json` (human-readable metadata).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"payload": payload, "meta": meta}, path)
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))


def load_processed(path):
    """Inverse of `save_processed`. Returns (payload, meta).

    `weights_only=False` is required because the payload contains PyG `Data` objects;
    only load files you produced yourself.
    """
    blob = torch.load(path, weights_only=False)
    return blob["payload"], blob["meta"]
