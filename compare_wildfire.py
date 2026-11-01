"""GCN vs MLP vs GBM on Next Day Wildfire Spread.

  python preprocess_wildfire.py --raw-dir data/raw/wildfire --out-dir data/processed/wildfire
  python compare_wildfire.py --task pixel
  python compare_wildfire.py --task patch

Tasks
  pixel   (node level)  will this 1 km pixel be burning tomorrow? Uncertain pixels (-1) are ignored.
                        AUPRC (primary), AUROC, F1 / precision / recall at the F1-optimal validation threshold.
  patch   (graph level) fraction of the patch burning tomorrow.                     MAE / RMSE / R2

What each model sees
  GCN        the pixel's 14 features + its 8-neighbour grid (messages from neighbouring pixels)
  MLP / GBM  the pixel's own 14 features only. The GCN therefore has spatial context the tabular models
             lack; --neighbor-baselines gives them [x, mean of the 8 neighbours' x] for a fairer reference
             (that baseline is effectively a one-layer convolution; a CNN is the natural next comparison).
  patch      MLP / GBM get mean/max/std of the pixel features over the patch; the GCN pools learned node states.

Cost control: tabular models train on --pixels-per-sample random labelled pixels from each training
patch; ALL labelled pixels of the first --max-eval-samples validation/test patches are scored by every
model. The GCN trains on whole patches (--max-train-samples). Class weights (capped at
--max-class-weight x) are the same for every model.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from compare_utils import (DEVICE, Results, Timer, add_model_args, balanced_class_weight,
                           constant_predictor, count_params, fit_gbm, fit_mlp, fit_torch,
                           neighbor_mean_features, score, selection_score, set_seed, to_scores)
from models import GraphGCN, NodeGCN
from preprocess_wildfire import WildfireGridDataset


# ------------------------------------ pixel task ------------------------------------
def pixel_table(ds, n, per_sample, rng, nbr):
    Xs, ys = [], []
    for i in range(min(n, len(ds))):
        d = ds[i]
        x = neighbor_mean_features(d.x, d.edge_index) if nbr else d.x
        idx = np.flatnonzero((d.y >= 0).numpy())
        if per_sample and len(idx) > per_sample:
            idx = rng.choice(idx, per_sample, replace=False)
        Xs.append(x.numpy()[idx])
        ys.append(d.y.numpy()[idx])
    return np.concatenate(Xs), np.concatenate(ys)


def gcn_pixel(args, dtr, dva, dte, w, seed):
    set_seed(seed)
    model = NodeGCN(dtr[0].x.size(1), args.hidden, 2, args.layers, args.dropout)
    wt = torch.from_numpy(w).to(DEVICE)
    ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)

    def train_epoch(opt):
        for b in ltr:
            b = b.to(DEVICE)
            loss = F.cross_entropy(model(b.x, b.edge_index), b.y, weight=wt, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def collect(ds):
        ys, ps = [], []
        for b in mk(ds):
            b = b.to(DEVICE)
            valid = b.y >= 0
            ps.append(F.softmax(model(b.x, b.edge_index), 1)[valid, 1])
            ys.append(b.y[valid])
        return torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()

    fit_torch(model, train_epoch, lambda: selection_score("binary", *collect(dva)), args.epochs,
              args.patience, args.lr, args.wd)
    (yv, pv), (yt, pt) = collect(dva), collect(dte)
    m = score("binary", yv, pv, yt, pt)
    return m, count_params(model)


def run_pixel(args, root, res):
    mk = lambda split: WildfireGridDataset(root, split, args.connectivity)
    ds_tr, ds_va, ds_te = mk("train"), mk("eval"), mk("test")
    n_tr, n_ev = min(args.max_train_samples, len(ds_tr)), args.max_eval_samples
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        Xtr, ytr = pixel_table(ds_tr, n_tr, args.pixels_per_sample, rng, False)
        Xva, yva = pixel_table(ds_va, n_ev, None, rng, False)
        Xte, yte = pixel_table(ds_te, n_ev, None, rng, False)
        w = balanced_class_weight(ytr, 2, args.max_class_weight)
        print(f"seed {seed}: train rows {len(ytr):,} (fire {ytr.mean():.2%}), eval rows {len(yva):,}/{len(yte):,}, "
              f"class weights {np.round(w, 2).tolist()}")

        with Timer() as t:
            m, n_par = gcn_pixel(args, ds_tr[:n_tr], ds_va[:n_ev], ds_te[:n_ev], w, seed)
        res.add("GCN", m, n_par, t.seconds)

        def tabular(name, tr_, va_, te_):
            ev = lambda pred: score("binary", va_[1], to_scores("binary", pred(va_[0])),
                                    te_[1], to_scores("binary", pred(te_[0])))
            with Timer() as t:
                pred, n_par = fit_mlp(*tr_, *va_, "binary", 2, args, w, seed)
            res.add(f"MLP{name}", ev(pred), n_par, t.seconds)
            with Timer() as t:
                pred = fit_gbm(*tr_, *va_, "binary", 2, args, w, seed)
            res.add(f"GBM{name}", ev(pred), None, t.seconds)

        tabular("", (Xtr, ytr), (Xva, yva), (Xte, yte))
        if args.neighbor_baselines:
            tabs = [pixel_table(ds_tr, n_tr, args.pixels_per_sample, np.random.default_rng(seed), True),
                    pixel_table(ds_va, n_ev, None, rng, True), pixel_table(ds_te, n_ev, None, rng, True)]
            tabular(" + nbr-mean", *tabs)
        c = constant_predictor("binary", ytr, 2)
        res.add("Constant (prior)", score("binary", yva, to_scores("binary", c(Xva)), yte,
                                          to_scores("binary", c(Xte))), None, 0.0)


# ------------------------------------ patch task ------------------------------------
def patch_table(ds, n):
    X = []
    for i in range(min(n, len(ds))):
        x = ds[i].x.numpy()
        X.append(np.concatenate([x.mean(0), x.max(0), x.std(0)]))
    return np.stack(X).astype(np.float32), ds.graph_y[:len(X), 0].astype(np.float32)


def gcn_patch(args, dtr, dva, dte, seed):
    set_seed(seed)
    model = GraphGCN(dtr[0].x.size(1), args.hidden, 1, args.layers, args.dropout, args.pooling)
    ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)

    def train_epoch(opt):
        for b in ltr:
            b = b.to(DEVICE)
            loss = F.mse_loss(model(b.x, b.edge_index, b.batch).squeeze(-1), b.y_graph)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def collect(ds):
        ys, ps = [], []
        for b in mk(ds):
            b = b.to(DEVICE)
            ps.append(model(b.x, b.edge_index, b.batch).squeeze(-1))
            ys.append(b.y_graph)
        return torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()

    fit_torch(model, train_epoch, lambda: selection_score("regression", *collect(dva)), args.epochs,
              args.patience, args.lr, args.wd)
    (yv, pv), (yt, pt) = collect(dva), collect(dte)
    return score("regression", yv, pv, yt, pt), count_params(model)


def run_patch(args, root, res):
    mk = lambda split: WildfireGridDataset(root, split, args.connectivity)
    ds_tr, ds_va, ds_te = mk("train"), mk("eval"), mk("test")
    n_tr, n_ev = min(args.max_train_samples, len(ds_tr)), args.max_eval_samples
    (Xtr, ytr), (Xva, yva), (Xte, yte) = patch_table(ds_tr, n_tr), patch_table(ds_va, n_ev), patch_table(ds_te, n_ev)
    print(f"patches train/val/test: {len(ytr)}/{len(yva)}/{len(yte)}")
    for seed in range(args.seeds):
        with Timer() as t:
            m, n_par = gcn_patch(args, ds_tr[:n_tr], ds_va[:n_ev], ds_te[:n_ev], seed)
        res.add("GCN", m, n_par, t.seconds)
        ev = lambda pred: score("regression", yva, pred(Xva), yte, pred(Xte))
        with Timer() as t:
            pred, n_par = fit_mlp(Xtr, ytr, Xva, yva, "regression", 1, args, None, seed)
        res.add("MLP", ev(pred), n_par, t.seconds)
        with Timer() as t:
            pred = fit_gbm(Xtr, ytr, Xva, yva, "regression", 1, args, None, seed)
        res.add("GBM", ev(pred), None, t.seconds)
        res.add("Constant (mean)", ev(constant_predictor("regression", ytr, 1)), None, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", default="data/processed/wildfire")
    p.add_argument("--task", choices=["pixel", "patch"], required=True)
    p.add_argument("--out", default=None, help="default: results/wildfire_<task>.json")
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    p.add_argument("--batch-size", type=int, default=16, help="patches per GCN batch")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--pixels-per-sample", type=int, default=256, help="tabular training pixels per patch")
    p.add_argument("--max-train-samples", type=int, default=10 ** 9)
    p.add_argument("--max-eval-samples", type=int, default=500, help="val/test patches scored (all models)")
    p.add_argument("--max-class-weight", type=float, default=20.0, help="cap on fire/no-fire weight ratio")
    p.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean", help="patch-level GCN readout")
    add_model_args(p)
    p.set_defaults(epochs=30, patience=5, lr=0.005, wd=1e-5, dropout=0.2, gbm_trees=300, mlp_batch=4096, seeds=1)
    args = p.parse_args()
    args.out = args.out or f"results/wildfire_{args.task}.json"

    res = Results(f"Next Day Wildfire Spread | task: {args.task}, {args.connectivity}-neighbour pixel grid")
    (run_pixel if args.task == "pixel" else run_patch)(args, args.root, res)
    print("\n" + res.table())
    res.save(args.out, args)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
