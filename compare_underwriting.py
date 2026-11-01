"""Graph neural networks vs GBM on the Belgian underwriting-fraud network: out-of-time node classification.

  python preprocess_underwriting.py --raw-dir data/raw/underwriting --out data/processed/underwriting.pt
  python compare_underwriting.py --target company --seeds 5 --neighbor-baselines
  python compare_underwriting.py --target admin --seeds 5 --neighbor-baselines

Setup (see preprocess_underwriting.py)
  train / val   labelled --target nodes of the t0 snapshot (stratified random split); every model is
                selected, early-stopped and thresholded on val only
  test          labelled --target nodes of the t1 snapshot that were not labelled at t0. The graph models
                are trained on the t0 graph and applied unchanged to the t1 graph (inductive, out of time).
What each model sees
  graph models  the whole heterogeneous neighbourhood (companies + administrators, node type as a feature).
                Plain GCN, plus "GCN + skip", "GraphSAGE" and "GAT" (graph attention) as separately named
                rows (--convs); all share width, depth, dropout, optimiser, learning rate, weight decay and
                early stopping.
  GBM           the target node's own features only. With --neighbor-baselines also "GBM + nbr-mean":
                [own features, mean of neighbours' features, log1p #neighbours per edge type], i.e.
                hand-engineered one-hop graph features, which separates "does the graph help?" from
                "does *learned* message passing help?".
  GBM + GAE     own features + unsupervised graph-autoencoder embeddings (skip with --no-gae): a GCN
                encoder trained without labels to reconstruct the t0 graph's links, then applied to t1.
Metrics on the test nodes: AUPRC (primary), AUROC, and F1 / accuracy at the validation-F1-optimal threshold.
The fraud rate shifts between snapshots (companies 49% -> 35%, administrators 19% -> 30%), which moves
AUPRC and the threshold-based metrics for every model alike.
"""
import argparse

import compare_utils  # noqa: F401  (must be first: imports LightGBM before PyTorch)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from compare_utils import (DEVICE, Results, Timer, add_gae_args, add_model_args, balanced_class_weight,
                           constant_predictor, count_params, fit_gbm, fit_torch, gae_embeddings, note_loss,
                           neighbor_mean_features, score, selection_score, set_seed, to_scores)
from graph_utils import load_processed
from models import NodeGCN

GCN_ROWS = {"gcn": "GCN", "gcn_skip": "GCN + skip", "sage": "GraphSAGE", "gat": "GAT"}
NAMES = {"company": "companies", "admin": "administrators"}


def run_gcn(args, d0, d1, target, w, seed, conv="gcn"):
    """Train on the t0 graph (val = t0 val nodes), then score the t1 test nodes with the same weights."""
    set_seed(seed)
    d0, d1 = d0.to(DEVICE), d1.to(DEVICE)
    tr, va, te = d0[f"train_mask_{target}"], d0[f"val_mask_{target}"], d1[f"test_mask_{target}"]
    model = NodeGCN(d0.x.size(1), args.hidden, 2, args.layers, args.dropout, conv=conv)
    wt = torch.from_numpy(w).to(DEVICE)
    y0, y1 = d0.y.cpu().numpy(), d1.y.cpu().numpy()

    def probs(d):
        return F.softmax(model(d.x, d.edge_index), 1)[:, 1].cpu().numpy()

    def train_epoch(opt):
        opt.zero_grad()
        loss = F.cross_entropy(model(d0.x, d0.edge_index)[tr], d0.y[tr], weight=wt)
        loss.backward()
        note_loss(loss)
        opt.step()

    va_np, te_np = va.cpu().numpy(), te.cpu().numpy()
    fit_torch(model, train_epoch, lambda: selection_score("binary", y0[va_np], probs(d0)[va_np]), args)
    with torch.no_grad():
        p0, p1 = probs(d0), probs(d1)
    return score("binary", y0[va_np], p0[va_np], y1[te_np], p1[te_np]), count_params(model)


def tables(d, own_idx, nbr):
    """Rows for the GBM: own features, optionally + neighbour means + log1p degree per edge type."""
    x = d.x[:, own_idx]
    if not nbr:
        return x.numpy()
    agg = neighbor_mean_features(d.x, d.edge_index)[:, d.x.size(1):]
    deg = scatter(F.one_hot(d.edge_type, 2).float(), d.edge_index[1], dim=0,
                  dim_size=d.num_nodes, reduce="sum")
    return torch.cat([x, agg, torch.log1p(deg)], 1).numpy()


def run_gbm(args, name, X0, X1, y0, y1, tr, va, te, w, seed, res, extra_seconds=0.0):
    ev = lambda pred: score("binary", y0[va], to_scores("binary", pred(X0[va])),
                            y1[te], to_scores("binary", pred(X1[te])))
    with Timer() as t:
        pred = fit_gbm(X0[tr], y0[tr], X0[va], y0[va], "binary", 2, args, w, seed)
    res.add(f"GBM{name}", ev(pred), None, t.seconds + extra_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/processed/underwriting.pt")
    p.add_argument("--target", choices=list(NAMES), default="company", help="which node type to classify")
    p.add_argument("--out", default=None, help="default: results/underwriting_<target>.json")
    p.add_argument("--convs", nargs="+", choices=list(GCN_ROWS), default=list(GCN_ROWS),
                   help="graph models to run, each its own row: plain GCN, GCN + skip connection "
                        "(own-feature term, see models.SkipGCNConv), GraphSAGE and graph attention (GAT)")
    add_model_args(p)
    add_gae_args(p)
    args = p.parse_args()
    args.out = args.out or f"results/underwriting_{args.target}.json"

    payload, meta = load_processed(args.data)
    d0, d1 = payload["snapshots"]["t0"], payload["snapshots"]["t1"]
    if args.layers >= meta["args"]["hops"]:
        raise SystemExit(f"--layers {args.layers} needs a graph preprocessed with --hops >= {args.layers + 1} "
                         f"(this one has {meta['args']['hops']})")
    t = args.target
    tr, va = d0[f"train_mask_{t}"].numpy(), d0[f"val_mask_{t}"].numpy()
    te = d1[f"test_mask_{t}"].numpy()
    y0, y1 = d0.y.numpy(), d1.y.numpy()
    w = balanced_class_weight(y0[tr], 2)
    own = meta[f"{t}_feature_idx"]
    X0, X1 = tables(d0, own, False), tables(d1, own, False)
    N0, N1 = (tables(d0, own, True), tables(d1, own, True)) if args.neighbor_baselines else (None, None)

    deg0 = torch.bincount(d0.edge_index[1], minlength=d0.num_nodes).numpy()
    print(f"t0 graph: {d0.num_nodes:,} nodes, {d0.edge_index.size(1) // 2:,} edges; {t} train nodes: median degree "
          f"{np.median(deg0[tr]):.0f}, without neighbours {np.mean(deg0[tr] == 0):.1%}")
    res = Results(f"Underwriting fraud: {NAMES[t]}, out-of-time (train t0 {tr.sum()}/val t0 {va.sum()}/"
                  f"test t1 {te.sum()}; fraud {y0[tr].mean():.0%} -> {y1[te].mean():.0%})")

    const = constant_predictor("binary", y0[tr], 2)
    for seed in range(args.seeds):
        print(f"seed {seed}: training {', '.join(GCN_ROWS[c] for c in args.convs)}, GBM, GAE ...", flush=True)
        for conv in args.convs:
            with Timer() as tm:
                m, n_par = run_gcn(args, d0.clone(), d1.clone(), t, w, seed, conv)
            res.add(GCN_ROWS[conv], m, n_par, tm.seconds)
        run_gbm(args, "", X0, X1, y0, y1, tr, va, te, w, seed, res)
        if N0 is not None:
            run_gbm(args, " + nbr-mean", N0, N1, y0, y1, tr, va, te, w, seed, res)
        if not args.no_gae:
            with Timer() as tm:
                z0, z1 = gae_embeddings(d0, args, seed, apply_to=(d1,))
            run_gbm(args, " + GAE", np.hstack([X0, z0]), np.hstack([X1, z1]), y0, y1, tr, va, te, w, seed, res,
                    tm.seconds)
        res.add("Constant (prior)", score("binary", y0[va], to_scores("binary", const(X0[va])),
                                          y1[te], to_scores("binary", const(X1[te]))), None, 0.0)
        print(f"seed {seed} done")
    print("\n" + res.table())
    res.save(args.out, args)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
