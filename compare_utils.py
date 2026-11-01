"""Shared pieces for the GCN vs MLP vs GBM comparison scripts.

Fairness rules every compare_*.py follows
  * identical train / val / test data for every model (splits come from the preprocessing step)
  * node-level tasks: MLP = the GCN with its graph convolutions replaced by plain Linear layers:
    same width, depth, dropout, optimiser, learning rate, weight decay, epochs and early stopping,
    so the only difference is message passing (parameter counts are equal, see the `params` column).
    Edge- and graph-level tasks use the same hyper-parameters, but the tabular models take pair /
    pooled features and have no GNN-specific heads, so their parameter counts differ.
  * GBM (LightGBM, or scikit-learn's HistGradientBoosting if LightGBM is missing) with early
    stopping on the same validation set
  * every model is selected on the validation set with the same criterion, and binary decision
    thresholds are tuned on validation only, never on test
  * class imbalance is handled identically (the same class weights in the loss / sample weights)
"""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from torch_geometric.utils import scatter

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
class MLP(nn.Module):
    """Same layout as NodeGCN/GCNEncoder with every GCNConv replaced by a Linear layer."""

    def __init__(self, in_ch, hidden, out_ch, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_ch] + [hidden] * (num_layers - 1) + [out_ch]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def fit_torch(model, train_epoch, val_score, epochs, patience, lr, weight_decay):
    """Generic early-stopped training loop. `train_epoch(opt)` runs one epoch; `val_score()` returns a
    float (higher is better). The best-validation weights are restored before returning."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        train_epoch(opt)
        model.eval()
        with torch.no_grad():
            s = val_score()
        if np.isfinite(s) and s > best:
            best, best_state, bad = s, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best


def fit_mlp(Xtr, ytr, Xva, yva, kind, num_classes, args, class_weight=None, seed=0):
    """Train an MLP on tabular rows. Returns (predict, n_params); predict(X) -> proba [n, C] or values [n]."""
    set_seed(seed)
    cls = kind != "regression"
    model = MLP(Xtr.shape[1], args.hidden, num_classes if cls else 1, args.layers, args.dropout)
    Xt = torch.from_numpy(np.asarray(Xtr, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(ytr)).long() if cls else torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    w = torch.from_numpy(class_weight).to(DEVICE) if (cls and class_weight is not None) else None
    bs = args.mlp_batch

    def predict(X):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                o = model(torch.from_numpy(np.asarray(X[i:i + 65536], dtype=np.float32)).to(DEVICE))
                outs.append(F.softmax(o, 1) if cls else o.squeeze(-1))
        return torch.cat(outs).cpu().numpy() if outs else np.zeros((0, num_classes) if cls else 0)

    def train_epoch(opt):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            out = model(xb)
            loss = F.cross_entropy(out, yb, weight=w) if cls else F.mse_loss(out.squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    fit_torch(model, train_epoch, lambda: selection_score(kind, yva, to_scores(kind, predict(Xva))),
              args.epochs, args.patience, args.lr, args.wd)
    return predict, count_params(model)


def resolve_gbm_backend(name):
    if name == "auto":
        try:
            import lightgbm  # noqa: F401
            return "lightgbm"
        except ImportError:
            return "sklearn"
    return name


def fit_gbm(Xtr, ytr, Xva, yva, kind, num_classes, args, class_weight=None, seed=0):
    """Gradient boosted trees with early stopping on the validation set. Returns predict(X)."""
    cls = kind != "regression"
    sw = class_weight[np.asarray(ytr, dtype=np.int64)] if (cls and class_weight is not None) else None
    backend = resolve_gbm_backend(args.gbm_backend)
    if backend == "lightgbm":
        import lightgbm as lgb
        params = dict(n_estimators=args.gbm_trees, learning_rate=args.gbm_lr, num_leaves=args.gbm_leaves,
                      min_child_samples=args.gbm_min_child, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
        model = lgb.LGBMClassifier(**params) if cls else lgb.LGBMRegressor(**params)
        model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(args.gbm_patience, verbose=False)])
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        params = dict(max_iter=args.gbm_trees, learning_rate=args.gbm_lr, max_leaf_nodes=args.gbm_leaves,
                      random_state=seed, early_stopping=False)
        model = HistGradientBoostingClassifier(**params) if cls else HistGradientBoostingRegressor(**params)
        model.fit(Xtr, ytr, sample_weight=sw)

    def predict(X):
        if not cls:
            return model.predict(X)
        proba = model.predict_proba(X)
        if proba.shape[1] < num_classes:           # a class absent from the training rows
            full = np.zeros((len(proba), num_classes))
            full[:, model.classes_.astype(int)] = proba
            return full
        return proba

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
        r = self.rows.setdefault(model, {"params": params, "runs": []})
        r["runs"].append({**metrics, "seconds": seconds})

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
    """CLI flags shared by every compare script (the GCN and MLP use the same ones)."""
    g = p.add_argument_group("shared model / training settings")
    g.add_argument("--hidden", type=int, default=64)
    g.add_argument("--layers", type=int, default=2)
    g.add_argument("--dropout", type=float, default=0.5)
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--wd", type=float, default=5e-4)
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--patience", type=int, default=30)
    g.add_argument("--mlp-batch", type=int, default=2048)
    g.add_argument("--seeds", type=int, default=3, help="repeat everything with seeds 0..N-1")
    g.add_argument("--gbm-backend", choices=["auto", "lightgbm", "sklearn"], default="auto")
    g.add_argument("--gbm-trees", type=int, default=500)
    g.add_argument("--gbm-lr", type=float, default=0.05)
    g.add_argument("--gbm-leaves", type=int, default=31)
    g.add_argument("--gbm-patience", type=int, default=30)
    g.add_argument("--gbm-min-child", type=int, default=20,
                   help="LightGBM min_child_samples; lower it (e.g. 3) for graph-level tasks with few snapshots")
    g.add_argument("--neighbor-baselines", action="store_true",
                   help="also run MLP/GBM on [x, mean of neighbours' x] (graph-aware tabular baseline)")
    return p
