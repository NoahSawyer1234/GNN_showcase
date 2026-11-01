"""GCN vs MLP vs GBM on Medicare provider fraud (node classification).

  python preprocess_medicare.py --raw-dir data/raw/medicare --out data/processed/medicare.pt
  python compare_medicare.py --data data/processed/medicare.pt --seeds 5 --neighbor-baselines

All three models see the same provider features and the same train/val/test providers. Only the
GCN also sees the provider-provider graph (shared patients / physicians). With --neighbor-baselines
the MLP and GBM additionally get [x, mean of neighbours' x], i.e. hand-engineered graph features,
which separates "does the graph help?" from "does *learned* message passing help?".
Metrics are on the test providers: AUPRC (primary, ~10% fraud), AUROC, and F1 / accuracy at the
threshold that maximises F1 on the validation providers.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from compare_utils import (DEVICE, Results, Timer, add_model_args, balanced_class_weight,
                           constant_predictor, count_params, fit_gbm, fit_mlp, fit_torch,
                           neighbor_mean_features, score, selection_score, set_seed, to_scores)
from graph_utils import load_processed
from models import NodeGCN


def run_gcn(args, data, w, seed):
    set_seed(seed)
    d = data.to(DEVICE)
    model = NodeGCN(d.x.size(1), args.hidden, 2, args.layers, args.dropout)
    wt = torch.from_numpy(w).to(DEVICE)
    y = d.y.cpu().numpy()
    va, te = d.val_mask.cpu().numpy(), d.test_mask.cpu().numpy()

    def forward():
        return model(d.x, d.edge_index, d.edge_weight)

    def probs():
        return F.softmax(forward(), 1)[:, 1].cpu().numpy()

    def train_epoch(opt):
        opt.zero_grad()
        F.cross_entropy(forward()[d.train_mask], d.y[d.train_mask], weight=wt).backward()
        opt.step()

    fit_torch(model, train_epoch, lambda: selection_score("binary", y[va], probs()[va]),
              args.epochs, args.patience, args.lr, args.wd)
    with torch.no_grad():
        p = probs()
    return score("binary", y[va], p[va], y[te], p[te]), count_params(model)


def run_tabular(args, name, X, y, tr, va, te, w, seed, res):
    ev = lambda pred: score("binary", y[va], to_scores("binary", pred(X[va])),
                            y[te], to_scores("binary", pred(X[te])))
    with Timer() as t:
        pred, n_par = fit_mlp(X[tr], y[tr], X[va], y[va], "binary", 2, args, w, seed)
    res.add(f"MLP{name}", ev(pred), n_par, t.seconds)
    with Timer() as t:
        pred = fit_gbm(X[tr], y[tr], X[va], y[va], "binary", 2, args, w, seed)
    res.add(f"GBM{name}", ev(pred), None, t.seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/processed/medicare.pt")
    p.add_argument("--out", default="results/medicare.json")
    add_model_args(p)
    args = p.parse_args()

    data, meta = load_processed(args.data)
    X, y = data.x.numpy(), data.y.numpy()
    tr, va, te = (m.numpy() for m in (data.train_mask, data.val_mask, data.test_mask))
    w = balanced_class_weight(y[tr], 2)
    Xn = neighbor_mean_features(data.x, data.edge_index, data.edge_weight).numpy() if args.neighbor_baselines else None
    res = Results(f"Medicare provider fraud: node classification (fraud rate {y.mean():.1%}, "
                  f"{tr.sum()}/{va.sum()}/{te.sum()} train/val/test providers)")

    const = constant_predictor("binary", y[tr], 2)
    for seed in range(args.seeds):
        with Timer() as t:
            m, n_par = run_gcn(args, data.clone(), w, seed)
        res.add("GCN", m, n_par, t.seconds)
        run_tabular(args, "", X, y, tr, va, te, w, seed, res)
        if Xn is not None:
            run_tabular(args, " + nbr-mean", Xn, y, tr, va, te, w, seed, res)
        res.add("Constant (prior)", score("binary", y[va], to_scores("binary", const(X[va])),
                                          y[te], to_scores("binary", const(X[te]))), None, 0.0)
        print(f"seed {seed} done")
    print("\n" + res.table())
    res.save(args.out, args)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
