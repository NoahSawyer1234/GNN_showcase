"""Shared pieces for the GCN vs GBM comparison scripts.

Fairness rules every compare_*.py follows
  * identical train / val / test data for every model (splits come from the preprocessing step)
  * every graph model (GCN and its named variants) uses the same width, depth, dropout, optimiser,
    learning rate, weight decay and early-stopping rule (fit_torch)
  * the baseline is a GBM (LightGBM, or scikit-learn's HistGradientBoosting if LightGBM is missing) on
    the same rows, with early stopping on the same validation set
  * every model is selected on the validation set with the same criterion, and binary decision
    thresholds are tuned on validation only, never on test
  * class imbalance is handled identically (the same class weights in the loss / sample weights)
"""
import platform

if platform.system() != "Darwin":
    try:  # Linux/Windows: import LightGBM before PyTorch so both OpenMP runtimes initialise cleanly.
        import lightgbm  # noqa: F401
    except ImportError:
        pass
# On macOS LightGBM is NOT imported here (and --gbm-backend auto uses scikit-learn): LightGBM and PyTorch
# ship different OpenMP runtimes there and loading both has caused segmentation faults in either order.
import atexit
import copy
import json
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from torch_geometric.utils import scatter

import gbm_worker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------- class weights -------------------------------
def balanced_class_weight(y, num_classes, max_ratio=None) -> np.ndarray:
    """n / (C * count_c); `max_ratio` caps the largest weight at max_ratio x the smallest."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    w = len(y) / (num_classes * np.clip(counts, 1, None))
    if max_ratio:
        w = np.minimum(w, w.min() * max_ratio)
    return w.astype(np.float32)


# ----------------------------------- metrics -----------------------------------
# kinds: "binary" (p = P(class 1)), "multiclass" (p = [n, C] probabilities),
#        "regression" (p = predicted values), "ranking" (p = any score; link prediction)
def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except (ValueError, IndexError):          # single-class or empty evaluation subsets -> undefined
        return float("nan")


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    # Midpoint between the best score and the next lower one: identical F1 on the validation set,
    # but less sensitive to exactly where the validation scores happen to fall.
    return float((thr[i] + thr[i - 1]) / 2) if i > 0 else float(thr[i])


def selection_score(kind, y, p) -> float:
    """Validation criterion used for early stopping / model selection (higher is better)."""
    if kind == "binary":
        return _safe(average_precision_score, y, p)
    if kind == "ranking":
        return _safe(roc_auc_score, y, p)
    if kind == "multiclass":
        return _safe(f1_score, y, np.argmax(p, 1), average="macro")
    return -_safe(mean_absolute_error, y, p)


def score(kind, y_val, p_val, y_test, p_test, test_subset=None) -> dict:
    """Test-set metrics. For binary tasks the F1 threshold is tuned on the validation set."""
    y_test, p_test = np.asarray(y_test), np.asarray(p_test)
    if kind == "binary":
        thr = best_f1_threshold(y_val, p_val)
        pred = p_test >= thr
        return {"auprc": _safe(average_precision_score, y_test, p_test),
                "auroc": _safe(roc_auc_score, y_test, p_test),
                "f1": _safe(f1_score, y_test, pred, zero_division=0),
                "accuracy": _safe(accuracy_score, y_test, pred)}
    if kind == "ranking":
        out = {"auroc": _safe(roc_auc_score, y_test, p_test),
               "ap": _safe(average_precision_score, y_test, p_test)}
        if test_subset is not None:      # e.g. only genuinely new links vs. non-lookback negatives
            s = np.asarray(test_subset, dtype=bool)
            out["auroc_new"] = _safe(roc_auc_score, y_test[s], p_test[s])
            out["ap_new"] = _safe(average_precision_score, y_test[s], p_test[s])
        return out
    if kind == "multiclass":
        pred = np.argmax(p_test, 1)
        return {"macro_f1": _safe(f1_score, y_test, pred, average="macro"),
                "accuracy": _safe(accuracy_score, y_test, pred)}
    return {"mae": _safe(mean_absolute_error, y_test, p_test),
            "rmse": float(np.sqrt(np.mean((y_test - p_test) ** 2))) if len(y_test) else float("nan"),
            "r2": _safe(r2_score, y_test, p_test)}


def to_scores(kind, proba):
    """Model output -> what the metrics expect (binary/ranking: P(class 1); else unchanged)."""
    return proba[:, 1] if kind in ("binary", "ranking") else proba


# ------------------------------------- models -------------------------------------
def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


LAST_FIT = None          # diagnostics of the most recent fit_torch call (picked up by Results.add)
_EPOCH_LOSSES = []


def note_loss(loss):
    """Called by every train_epoch after each backward(): collects the training loss for the plateau rule."""
    _EPOCH_LOSSES.append(float(loss.detach()))


def fit_torch(model, train_epoch, val_score, args):
    """Early-stopped training loop shared by every graph model (so all get the identical rule).

    One epoch = `train_epoch(opt)`; `val_score()` returns a float, higher is better. Training stops at the
    FIRST of:
      1. validation score has not improved by > --min-delta for --patience epochs   (the main rule)
      2. mean training loss has not improved by > 0.1% (relative) for --loss-patience epochs
      3. --epochs (a safety cap only; the run is flagged 'hit cap' if it gets there)
    The learning rate is halved (--lr-decay) after --patience/3 epochs without validation improvement, which
    lets a model keep improving with finer steps before rule 1 fires. The best-validation weights are
    restored before returning, and the diagnostics are stored in LAST_FIT.
    """
    global LAST_FIT
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = None
    if args.lr_decay < 1.0:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=args.lr_decay,
                                                           patience=max(1, args.patience // 3), min_lr=args.lr * 1e-3)
    best, best_epoch, best_state, bad = -np.inf, 0, None, 0
    best_loss, loss_bad, reason, epoch = np.inf, 0, "hit cap", 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        _EPOCH_LOSSES.clear()
        train_epoch(opt)
        tl = float(np.mean(_EPOCH_LOSSES)) if _EPOCH_LOSSES else np.nan
        model.eval()
        with torch.no_grad():
            s = val_score()
        if sched is not None and np.isfinite(s):
            sched.step(s)
        if np.isfinite(s) and s > best + args.min_delta:
            best, best_epoch, bad = s, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                reason = "val patience"
                break
        if np.isfinite(tl):
            if tl < best_loss * (1 - 1e-3):
                best_loss, loss_bad = tl, 0
            else:
                loss_bad += 1
                if args.loss_patience > 0 and loss_bad >= args.loss_patience:
                    reason = "train-loss plateau"
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    LAST_FIT = {"epochs": epoch, "best_epoch": best_epoch, "stopped_by": reason,
                "final_lr": float(opt.param_groups[0]["lr"]), "best_val": float(best)}
    return best


def resolve_gbm_backend(name):
    if name == "auto":
        import platform
        if platform.system() == "Darwin":      # LightGBM + PyTorch have segfaulted on macOS
            return "sklearn"                   # (force LightGBM there with --gbm-backend lightgbm)
        try:
            import lightgbm  # noqa: F401
            return "lightgbm"
        except ImportError:
            return "sklearn"
    return name


def resolve_gbm_isolation(name):
    """Run the GBM in a separate process? auto = yes on macOS (PyTorch + OpenMP segfaults), no elsewhere."""
    return name == "yes" or (name == "auto" and platform.system() == "Darwin")


def _run_gbm_worker(mode, workdir):
    cmd = [sys.executable, str(Path(__file__).with_name("gbm_worker.py")), mode, str(workdir)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"GBM subprocess '{mode}' failed with exit code {r.returncode}. "
                           "Try --gbm-backend sklearn, or --gbm-isolate no if you are not on macOS.")


def fit_gbm(Xtr, ytr, Xva, yva, kind, num_classes, args, class_weight=None, seed=0):
    """Gradient boosted trees with early stopping on the validation set. Returns predict(X).

    Optionally runs in a separate process (see gbm_worker.py) so PyTorch is never loaded alongside it."""
    job = dict(Xtr=np.asarray(Xtr, dtype=np.float32), ytr=np.asarray(ytr),
               Xva=np.asarray(Xva, dtype=np.float32), yva=np.asarray(yva),
               cls=kind != "regression", num_classes=num_classes, class_weight=class_weight,
               backend=resolve_gbm_backend(args.gbm_backend), trees=args.gbm_trees, lr=args.gbm_lr,
               leaves=args.gbm_leaves, min_child=args.gbm_min_child, patience=args.gbm_patience, seed=seed)
    if not resolve_gbm_isolation(args.gbm_isolate):
        obj = gbm_worker.fit(job)
        return lambda X: gbm_worker.predict(obj, X)
    tmp = Path(tempfile.mkdtemp(prefix="gbm_"))
    atexit.register(shutil.rmtree, tmp, True)
    with open(tmp / "job.pkl", "wb") as f:
        pickle.dump(job, f, protocol=4)
    _run_gbm_worker("fit", tmp)

    def predict(X):
        np.save(tmp / "X.npy", np.ascontiguousarray(X, dtype=np.float32))
        _run_gbm_worker("predict", tmp)
        return np.load(tmp / "out.npy")

    return predict


def constant_predictor(kind, ytr, num_classes):
    """Reference row: predicts the training prior (classification) or training mean (regression)."""
    if kind == "regression":
        m = float(np.mean(ytr))
        return lambda X: np.full(len(X), m)
    prior = np.bincount(np.asarray(ytr, dtype=np.int64), minlength=num_classes) / len(ytr)
    return lambda X: np.tile(prior, (len(X), 1))


def neighbor_mean_features(x, edge_index, edge_weight=None) -> torch.Tensor:
    """[x, (weighted) mean of neighbours' x]: a hand-engineered one-hop aggregation for tabular models."""
    x = torch.as_tensor(x, dtype=torch.float32)
    src, dst = edge_index
    w = torch.ones(src.numel()) if edge_weight is None else torch.as_tensor(edge_weight, dtype=torch.float32)
    agg = scatter(x[src] * w[:, None], dst, dim=0, dim_size=x.size(0), reduce="sum")
    den = scatter(w, dst, dim=0, dim_size=x.size(0), reduce="sum")
    return torch.cat([x, agg / den.clamp(min=1e-9)[:, None]], dim=1)


# ------------------------------------- reporting -------------------------------------
class Results:
    """Collects per-seed metrics for each model and prints mean +- std."""

    def __init__(self, title):
        self.title, self.rows = title, {}

    def add(self, model, metrics, params=None, seconds=None):
        global LAST_FIT
        r = self.rows.setdefault(model, {"params": params, "runs": []})
        r["runs"].append({**metrics, "seconds": seconds})
        note = ""
        if LAST_FIT is not None:                       # a graph model was just trained
            r.setdefault("fit", []).append(LAST_FIT)
            note = (f"; {LAST_FIT['epochs']} epochs, best at {LAST_FIT['best_epoch']}, "
                    f"stopped by {LAST_FIT['stopped_by']}")
            if LAST_FIT["stopped_by"] == "hit cap":
                note += "  ** still improving at the cap: raise --epochs **"
            LAST_FIT = None
        print(f"  [{model}] finished" + (f" in {seconds:.1f}s" if seconds else "") + note, flush=True)

    def summary(self):
        out = {}
        for m, r in self.rows.items():
            keys = [k for k in r["runs"][0]]
            out[m] = {k: (float(np.nanmean([x[k] for x in r["runs"]])),
                          float(np.nanstd([x[k] for x in r["runs"]]))) for k in keys}
        return out

    def table(self):
        s = self.summary()
        keys = [k for k in next(iter(s.values())) if k != "seconds"]
        n = len(next(iter(self.rows.values()))["runs"])
        head = ["model"] + keys + ["params", "sec"]
        lines = []
        for m, v in s.items():
            cells = [m]
            for k in keys:
                mu, sd = v[k]
                cells.append(f"{mu:.4f}" + (f" ±{sd:.4f}" if n > 1 else ""))
            p = self.rows[m]["params"]
            cells += [f"{p:,}" if p else "-", f"{v['seconds'][0]:.1f}"]
            lines.append(cells)
        width = [max(len(r[i]) for r in [head] + lines) for i in range(len(head))]
        fmt = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, width))
        return "\n".join([f"{self.title}   ({n} seed{'s' if n > 1 else ''}, test set)", fmt(head),
                          "  ".join("-" * w for w in width)] + [fmt(r) for r in lines])

    def save(self, path, args):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"title": self.title, "args": vars(args), "summary": self.summary(),
                                    "runs": self.rows}, indent=2, default=str))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0


def add_model_args(p):
    """CLI flags shared by every compare script (every graph model uses the same ones)."""
    g = p.add_argument_group("shared model / training settings")
    g.add_argument("--hidden", type=int, default=64)
    g.add_argument("--layers", type=int, default=2)
    g.add_argument("--dropout", type=float, default=0.5)
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--wd", type=float, default=5e-4)
    g.add_argument("--epochs", type=int, default=1000, help="safety cap only; early stopping normally ends training")
    g.add_argument("--patience", type=int, default=50, help="stop after this many epochs without validation improvement")
    g.add_argument("--min-delta", type=float, default=1e-4, help="smallest validation improvement that counts")
    g.add_argument("--loss-patience", type=int, default=100,
                   help="also stop if mean training loss has not improved for this many epochs (0 = off)")
    g.add_argument("--lr-decay", type=float, default=0.5,
                   help="halve the LR after patience/3 epochs without validation improvement (1.0 = off)")
    g.add_argument("--seeds", type=int, default=3, help="repeat everything with seeds 0..N-1")
    g.add_argument("--gbm-backend", choices=["auto", "lightgbm", "sklearn"], default="auto",
                   help="auto = LightGBM, except scikit-learn on macOS")
    g.add_argument("--gbm-isolate", choices=["auto", "yes", "no"], default="auto",
                   help="run the GBM in its own process (auto = yes on macOS; avoids PyTorch/OpenMP segfaults)")
    g.add_argument("--gbm-trees", type=int, default=500)
    g.add_argument("--gbm-lr", type=float, default=0.05)
    g.add_argument("--gbm-leaves", type=int, default=31)
    g.add_argument("--gbm-patience", type=int, default=30)
    g.add_argument("--gbm-min-child", type=int, default=20,
                   help="LightGBM min_child_samples; lower it (e.g. 3) for graph-level tasks with few snapshots")
    g.add_argument("--neighbor-baselines", action="store_true",
                   help="also run the GBM on [x, mean of neighbours' x] (graph-aware tabular baseline)")
    return p
