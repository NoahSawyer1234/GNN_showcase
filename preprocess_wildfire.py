"""Next Day Wildfire Spread (Huot et al.) TFRecords -> grid graphs, one graph per 64x64 patch.

  python preprocess_wildfire.py --raw-dir data/raw/wildfire --out-dir data/processed/wildfire

Download the TFRecords yourself (Kaggle: "Next Day Wildfire Spread"); expected file names look like
next_day_wildfire_spread_{train,eval,test}_NN.tfrecord (plain or gzip). No TensorFlow needed.

Graph definition (per sample)
  nodes   the 4096 pixels (1 km each), node id = row * 64 + col
  edges   grid adjacency, 8-neighbour by default (--connectivity 4 for von Neumann);
          identical for every sample, so it is stored ONCE and shared
  x       [4096, 14] standardised features (see FEATURE_NAMES; wind direction -> sin/cos,
          PrevFireMask split into a binary flag + an 'unknown' flag)
  y       [4096] next-day fire mask: 1 fire, 0 no fire, -1 uncertain (cloud / missing -> ignore in loss)
  y_graph [1] fraction of labelled pixels burning next day (graph-level regression)
  y_graph_cls [1] 1 if the burning-pixel count grows vs. the previous day (graph-level classification)

Storage: per split `<split>_x.npy` (N, 4096, F), `<split>_y.npy` (N, 4096) int8,
`<split>_graph_y.npy` (N, 2) plus `meta.json`; read them with `WildfireGridDataset`, which builds
PyG `Data` objects on the fly from memory-mapped arrays (the full train set is several GB as Data).

Feature statistics (outlier clip bounds, mean, std) come from a sample of the TRAIN split only.
NaN inputs are replaced by the training mean (= 0 after standardisation); NaN rates are in meta.json.
"""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

RAW_INPUTS = ["elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI",
              "population", "erc", "PrevFireMask"]
CONT = ["elevation", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc"]
FEATURE_NAMES = CONT + ["th_sin", "th_cos", "PrevFireMask", "PrevFireMask_unknown"]
SPLITS = ("train", "eval", "test")


# ------------------------- minimal TFRecord / tf.Example reader -------------------------
def _open(path):
    path = str(path)
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if gz else open(path, "rb")


def iter_tfrecord(path):
    """Yield the raw bytes of each record (CRCs are not verified)."""
    with _open(path) as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (length,) = struct.unpack("<Q", head[:8])
            data = f.read(length)
            f.read(4)
            yield data


def count_tfrecord(path) -> int:
    n = 0
    with _open(path) as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return n
            (length,) = struct.unpack("<Q", head[:8])
            f.seek(length + 4, 1)
            n += 1


def _varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf):
    """Iterate protobuf fields of a message: yields (field_number, wire_type, value)."""
    buf = memoryview(buf)
    pos, n = 0, len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _varint(buf, pos)
        elif wt == 2:
            ln, pos = _varint(buf, pos)
            v, pos = buf[pos:pos + ln], pos + ln
        elif wt == 5:
            v, pos = buf[pos:pos + 4], pos + 4
        elif wt == 1:
            v, pos = buf[pos:pos + 8], pos + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield fno, wt, v


def _parse_feature(buf):
    for fno, _, v in _fields(buf):
        if fno == 2:                                          # FloatList
            parts = []
            for f2, wt2, v2 in _fields(v):
                if f2 == 1:
                    parts.append(np.frombuffer(v2, dtype="<f4") if wt2 == 2
                                 else np.frombuffer(v2, dtype="<f4", count=1))
            return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
        if fno == 3:                                          # Int64List
            vals = []
            for f2, wt2, v2 in _fields(v):
                if f2 == 1 and wt2 == 2:
                    pos = 0
                    while pos < len(v2):
                        x, pos = _varint(v2, pos)
                        vals.append(x - (1 << 64) if x >= 1 << 63 else x)
                elif f2 == 1:
                    vals.append(v2 - (1 << 64) if v2 >= 1 << 63 else v2)
            return np.asarray(vals, dtype=np.float32)
    return None


def parse_example(buf) -> dict:
    """tf.train.Example bytes -> {feature_name: float32 array}."""
    out = {}
    for fno, _, features in _fields(buf):
        if fno != 1:
            continue
        for f2, _, entry in _fields(features):
            if f2 != 1:
                continue
            key = val = None
            for f3, _, v3 in _fields(entry):
                if f3 == 1:
                    key = bytes(v3).decode()
                elif f3 == 2:
                    val = _parse_feature(v3)
            if key is not None and val is not None:
                out[key] = val
    return out


def split_files(raw_dir, split):
    return sorted(p for p in Path(raw_dir).glob(f"*{split}*.tfrecord*") if p.is_file())


def iter_split(raw_dir, split, max_samples=None):
    n = 0
    for f in split_files(raw_dir, split):
        for rec in iter_tfrecord(f):
            if max_samples is not None and n >= max_samples:
                return
            yield parse_example(rec)
            n += 1


def count_split(raw_dir, split, max_samples=None):
    n = sum(count_tfrecord(f) for f in split_files(raw_dir, split))
    return n if max_samples is None else min(n, max_samples)


# ------------------------------- feature construction -------------------------------
def _get(ex, key, size):
    if key not in ex:
        raise KeyError(f"Feature '{key}' not found in TFRecord (have: {sorted(ex)})")
    arr = ex[key]
    if arr.size != size:
        raise ValueError(f"Feature '{key}' has {arr.size} values, expected {size}")
    return arr.astype(np.float32)


def raw_arrays(ex, side):
    size = side * side
    return {k: _get(ex, k, size) for k in RAW_INPUTS + ["FireMask"]}


def build_features(raw, stats, dtype):
    cont = np.stack([raw[k] for k in CONT], axis=1)
    cont = np.clip(cont, stats["lo"], stats["hi"])
    cont = np.nan_to_num((cont - stats["mean"]) / stats["std"], nan=0.0)
    theta = np.deg2rad(raw["th"])                              # degrees clockwise from north
    prev = raw["PrevFireMask"]
    unknown = (~np.isfinite(prev)) | (prev < 0)
    cols = [cont, np.nan_to_num(np.sin(theta))[:, None], np.nan_to_num(np.cos(theta))[:, None],
            (np.nan_to_num(prev) > 0).astype(np.float32)[:, None], unknown.astype(np.float32)[:, None]]
    return np.concatenate(cols, axis=1).astype(dtype)


def build_labels(raw):
    fire = raw["FireMask"]
    valid = np.isfinite(fire) & (fire >= 0)
    y = np.full(fire.shape, -1, dtype=np.int8)
    y[valid] = (fire[valid] > 0).astype(np.int8)
    n_next, n_valid = int((y == 1).sum()), int(valid.sum())
    n_prev = int((np.nan_to_num(raw["PrevFireMask"]) > 0).sum())
    return y, np.array([n_next / max(n_valid, 1), float(n_next > n_prev)], dtype=np.float32)


def compute_stats(raw_dir, n_train, side, n_sample=1000, max_samples=None):
    """Clip bounds (p1/p99) and mean/std of the continuous channels from a train subsample."""
    stride = max(1, n_train // n_sample)
    rows = []
    for i, ex in enumerate(iter_split(raw_dir, "train", max_samples)):
        if i % stride == 0:
            raw = raw_arrays(ex, side)
            rows.append(np.stack([raw[k] for k in CONT], axis=1))
    arr = np.concatenate(rows)                                  # (samples * pixels, channels)
    lo, hi = np.nanpercentile(arr, [1, 99], axis=0)
    clipped = np.clip(arr, lo, hi)
    std = np.nanstd(clipped, axis=0)
    return {"lo": lo.astype(np.float32), "hi": hi.astype(np.float32),
            "mean": np.nanmean(clipped, axis=0).astype(np.float32),
            "std": np.where(std < 1e-6, 1.0, std).astype(np.float32)}


# ----------------------------------------- main -----------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/wildfire")
    p.add_argument("--out-dir", default="data/processed/wildfire")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--max-samples", type=int, default=None, help="debug: cap samples per split")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {s: count_split(args.raw_dir, s, args.max_samples) for s in SPLITS}
    if counts["train"] == 0:
        raise FileNotFoundError(f"No train TFRecords found in {args.raw_dir} "
                                "(expected files like next_day_wildfire_spread_train_00.tfrecord)")
    first = next(iter_split(args.raw_dir, "train", 1))
    side = int(round(np.sqrt(first["elevation"].size)))
    print(f"patch {side}x{side}; samples: {counts}")

    stats = compute_stats(args.raw_dir, counts["train"], side, max_samples=args.max_samples)
    nan_frac = np.zeros(len(CONT))
    n_pix_seen = 0
    for split in SPLITS:
        n = counts[split]
        if n == 0:
            print(f"  (no records for split '{split}', skipping)")
            continue
        x = np.lib.format.open_memmap(out / f"{split}_x.npy", mode="w+", dtype=args.dtype,
                                      shape=(n, side * side, len(FEATURE_NAMES)))
        y = np.lib.format.open_memmap(out / f"{split}_y.npy", mode="w+", dtype=np.int8,
                                      shape=(n, side * side))
        gy = np.zeros((n, 2), dtype=np.float32)
        for i, ex in enumerate(iter_split(args.raw_dir, split, args.max_samples)):
            raw = raw_arrays(ex, side)
            if split == "train":
                nan_frac += np.isnan(np.stack([raw[k] for k in CONT], 1)).sum(0)
                n_pix_seen += side * side
            x[i] = build_features(raw, stats, args.dtype)
            y[i], gy[i] = build_labels(raw)
        x.flush(); y.flush()
        np.save(out / f"{split}_graph_y.npy", gy)
        valid = (y[:] >= 0)
        print(f"  {split}: x{tuple(x.shape)}  fire pixels: {(y[:] == 1).sum() / max(valid.sum(), 1):.3%} "
              f"of labelled, uncertain: {1 - valid.mean():.2%}, graphs growing: {gy[:, 1].mean():.1%}")

    meta = {"dataset": "next_day_wildfire_spread", "side": side, "feature_names": FEATURE_NAMES,
            "continuous_channels": CONT, "dtype": args.dtype, "num_samples": counts,
            "stats": {k: v.tolist() for k, v in stats.items()},
            "train_nan_fraction": dict(zip(CONT, (nan_frac / max(n_pix_seen, 1)).tolist())),
            "graph_targets": ["burning_fraction_next_day", "fire_count_grew"],
            "label_values": {"1": "fire", "0": "no fire", "-1": "uncertain (ignore)"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {out}")


# ------------------------------------ torch dataset ------------------------------------
def grid_edge_index(h: int, w: int, connectivity: int = 8) -> torch.Tensor:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    src, dst = [], []
    for dr, dc in offsets:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        src.append((rr * w + cc)[ok])
        dst.append((r2 * w + c2)[ok])
    return torch.from_numpy(np.stack([np.concatenate(src), np.concatenate(dst)])).long()


class WildfireGridDataset(Dataset):
    """One PyG `Data` per patch, built lazily from the arrays written by this script.

        ds = WildfireGridDataset("data/processed/wildfire", split="train")
        loader = torch_geometric.loader.DataLoader(ds, batch_size=16, shuffle=True)
        # batch.x [B*4096, F], batch.y [B*4096] (-1 = ignore), batch.y_graph [B], batch.y_graph_cls [B]
    """

    def __init__(self, root, split="train", connectivity=8, transform=None):
        super().__init__(root=None, transform=transform)
        self.dir = Path(root)
        self.split = split
        self.meta = json.loads((self.dir / "meta.json").read_text())
        side = self.meta["side"]
        self.edge_index = grid_edge_index(side, side, connectivity)    # shared by every sample
        self.graph_y = np.load(self.dir / f"{split}_graph_y.npy")
        self._x = self._y = None                                       # opened lazily (worker-safe)

    @property
    def feature_names(self):
        return self.meta["feature_names"]

    def len(self):
        return len(self.graph_y)

    def get(self, idx):
        if self._x is None:
            self._x = np.load(self.dir / f"{self.split}_x.npy", mmap_mode="r")
            self._y = np.load(self.dir / f"{self.split}_y.npy", mmap_mode="r")
        gy = self.graph_y[idx]
        return Data(x=torch.from_numpy(np.asarray(self._x[idx], dtype=np.float32)),
                    edge_index=self.edge_index,
                    y=torch.from_numpy(np.asarray(self._y[idx], dtype=np.int64)),
                    y_graph=torch.tensor([gy[0]], dtype=torch.float32),
                    y_graph_cls=torch.tensor([int(gy[1])], dtype=torch.long))


if __name__ == "__main__":
    main()
