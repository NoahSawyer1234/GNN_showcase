"""GCN vs MLP vs GBM on GDELT country-interaction snapshots, for node-, edge- and graph-level tasks.

  python preprocess_gdelt.py --raw-dir data/raw/gdelt --out data/processed/gdelt.pt
  python compare_gdelt.py --task node_class
  python compare_gdelt.py --task link_prediction --seeds 5

Tasks (every snapshot: inputs from the previous window, targets from the current window)
  node_class       dominant event type (QuadClass) of each country         macro-F1 / accuracy
  node_reg         --reg-target (goldstein | tone | conflict_share)         MAE / RMSE / R2
  link_prediction  will a country pair interact?  positives = pairs with events in the window,
                   negatives = equal number of random non-interacting pairs        AUROC / AP
  edge_class       dominant event type of an interacting pair                      macro-F1 / accuracy
  graph_reg        --graph-target of the whole window (worldwide)                  MAE / RMSE / R2
  graph_cls        is the window's conflict share above the training median?       AUROC / AUPRC / F1

What each model sees
  GCN        node features + the lookback interaction graph (weighted edges)
  MLP / GBM  node tasks: the node's own features.   edge tasks: symmetric features of the two
             endpoints [xi+xj, |xi-xj|, xi*xj] plus the pair's own lookback interaction history
             (turn off with --no-pair-history).      graph tasks: mean/max/std of node features
             + edge count/density.   (--neighbor-baselines: node tasks only, adds neighbour means)
`edge_*` metrics with the `_new` suffix (link prediction) only count genuinely NEW links against
negatives that also did not interact in the lookback window, which removes the "same as yesterday" shortcut.
Splits are chronological (from preprocess_gdelt.py); with few snapshots, graph-level numbers are noisy.
"""
import argparse

import compare_utils  # noqa: F401  (must be first: imports LightGBM before PyTorch)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from compare_utils import (DEVICE, Results, Timer, add_model_args, balanced_class_weight,
                           constant_predictor, count_params, fit_gbm, fit_mlp, fit_torch,
                           neighbor_mean_features, score, selection_score, set_seed, to_scores)
from graph_utils import load_processed
from models import EdgeGCN, GraphGCN, NodeGCN

TASKS = ["node_class", "node_reg", "link_prediction", "edge_class", "graph_reg", "graph_cls"]
REG_TARGETS = ["goldstein", "tone", "conflict_share"]


# ------------------------------ tabular feature builders ------------------------------
def node_table(snaps, task, ridx, nbr):
    Xs, ys = [], []
    for d in snaps:
        m = d.node_target_mask
        x = neighbor_mean_features(d.x, d.edge_index, d.edge_weight) if nbr else d.x
        y = d.y_node_cls if task == "node_class" else d.y_node_reg[:, ridx]
        Xs.append(x[m].numpy())
        ys.append(y[m].numpy())
    return np.concatenate(Xs), np.concatenate(ys)


def lookback_keys(d):
    ei = d.edge_index.numpy()
    m = ei[0] < ei[1]
    return ei[0][m].astype(np.int64) * d.num_nodes + ei[1][m]


def pair_features(d, pairs, history=True):
    x = d.x.numpy()
    i, j = pairs
    f = [x[i] + x[j], np.abs(x[i] - x[j]), x[i] * x[j]]
    if history:
        ei = d.edge_index.numpy()
        m = ei[0] < ei[1]
        keys = ei[0][m].astype(np.int64) * d.num_nodes + ei[1][m]
        order = np.argsort(keys)
        keys, attr = keys[order], d.edge_attr.numpy()[m][order]
        q = np.minimum(i, j).astype(np.int64) * d.num_nodes + np.maximum(i, j)
        if len(keys):
            pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
            found = keys[pos] == q
            h = np.where(found[:, None], attr[pos], 0.0)
        else:
            found, h = np.zeros(len(q), bool), np.zeros((len(q), d.edge_attr.size(1)))
        f += [h, found[:, None].astype(np.float32)]
    return np.concatenate(f, 1).astype(np.float32)


def add_negatives(d, rng):
    """Attach as many random non-interacting pairs (lo<hi) as there are positives to snapshot `d`."""
    n = d.num_nodes
    pos = d.edge_label_index.numpy()
    n_pos = pos.shape[1]
    pos_keys = pos[0].astype(np.int64) * n + pos[1]
    got = np.zeros(0, np.int64)
    for _ in range(20):
        if len(got) >= n_pos:
            break
        a, b = rng.integers(0, n, 4 * n_pos + 16), rng.integers(0, n, 4 * n_pos + 16)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        keys = np.unique(lo[lo != hi].astype(np.int64) * n + hi[lo != hi])
        got = np.unique(np.concatenate([got, keys[~np.isin(keys, pos_keys)]]))
    got = rng.permutation(got)[:n_pos]
    d.neg_edge_label_index = torch.from_numpy(np.stack([got // n, got % n]).astype(np.int64))
    d.neg_in_lookback = torch.from_numpy(np.isin(got, lookback_keys(d)))


def lp_table(snaps, history):
    Xs, ys, new = [], [], []
    for d in snaps:
        for pairs, lab, nw in ((d.edge_label_index.numpy(), 1, d.edge_label_is_new.numpy()),
                               (d.neg_edge_label_index.numpy(), 0, ~d.neg_in_lookback.numpy())):
            Xs.append(pair_features(d, pairs, history))
            ys.append(np.full(pairs.shape[1], lab))
            new.append(nw)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(new)


def edge_class_table(snaps, history):
    return (np.concatenate([pair_features(d, d.edge_label_index.numpy(), history) for d in snaps]),
            np.concatenate([d.edge_label_class.numpy() for d in snaps]))


def graph_table(snaps, task, gidx):
    X = []
    for d in snaps:
        x, n_e = d.x.numpy(), d.edge_index.size(1) / 2
        X.append(np.concatenate([x.mean(0), x.max(0), x.std(0),
                                 [np.log1p(n_e), n_e / max(d.num_nodes * (d.num_nodes - 1) / 2, 1)]]))
    y = (np.concatenate([d.y_graph_cls.numpy() for d in snaps]) if task == "graph_cls"
         else np.array([d.y_graph[0, gidx].item() for d in snaps]))
    return np.stack(X).astype(np.float32), y


# ------------------------------------- GCN runners -------------------------------------
def loaders(args, tr, va, te):
    mk = lambda s, sh: DataLoader(s, batch_size=args.batch_size, shuffle=sh)
    return mk(tr, True), mk(va, False), mk(te, False)


def gcn_node(args, tr, va, te, task, ridx, w, C, seed):
    set_seed(seed)
    cls = task == "node_class"
    kind = "multiclass" if cls else "regression"
    model = NodeGCN(tr[0].x.size(1), args.hidden, C if cls else 1, args.layers, args.dropout)
    wt = torch.from_numpy(w).to(DEVICE) if cls else None
    ltr, lva, lte = loaders(args, tr, va, te)

    def out_y(b):
        out = model(b.x, b.edge_index, b.edge_weight)[b.node_target_mask]
        y = b.y_node_cls[b.node_target_mask] if cls else b.y_node_reg[b.node_target_mask][:, ridx]
        return out, y

    def train_epoch(opt):
        for b in ltr:
            b = b.to(DEVICE)
            out, y = out_y(b)
            if len(y) == 0:
                continue
            loss = F.cross_entropy(out, y, weight=wt) if cls else F.mse_loss(out.squeeze(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def collect(loader):
        ys, ps = [], []
        for b in loader:
            out, y = out_y(b.to(DEVICE))
            ps.append(F.softmax(out, 1) if cls else out.squeeze(-1))
            ys.append(y)
        return torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()

    fit_torch(model, train_epoch, lambda: selection_score(kind, *collect(lva)), args.epochs,
              args.patience, args.lr, args.wd)
    (yv, pv), (yt, pt) = collect(lva), collect(lte)
    return score(kind, yv, pv, yt, pt), count_params(model)


def gcn_link(args, tr, va, te, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = EdgeGCN(tr[0].x.size(1), args.hidden, args.embed, args.layers, args.dropout)
    ltr, lva, lte = loaders(args, tr, va, te)

    def train_epoch(opt):
        for d in tr:
            add_negatives(d, rng)                       # fresh negatives every epoch
        for b in ltr:
            b = b.to(DEVICE)
            z = model.encode(b.x, b.edge_index, b.edge_weight)
            pos = model.decode_link(z, b.edge_label_index)
            neg = model.decode_link(z, b.neg_edge_label_index)
            loss = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                                                      torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def collect(loader):
        ys, ss, nw = [], [], []
        for b in loader:
            b = b.to(DEVICE)
            z = model.encode(b.x, b.edge_index, b.edge_weight)
            ss += [model.decode_link(z, b.edge_label_index), model.decode_link(z, b.neg_edge_label_index)]
            ys += [torch.ones(b.edge_label_index.size(1)), torch.zeros(b.neg_edge_label_index.size(1))]
            nw += [b.edge_label_is_new.cpu(), ~b.neg_in_lookback.cpu()]
        return (torch.cat(ys).numpy(), torch.cat(ss).cpu().numpy(), torch.cat(nw).numpy())

    fit_torch(model, train_epoch, lambda: selection_score("ranking", *collect(lva)[:2]), args.epochs,
              args.patience, args.lr, args.wd)
    (yv, sv, _), (yt, st, nt) = collect(lva), collect(lte)
    return score("ranking", yv, sv, yt, st, test_subset=nt), count_params(model)


def gcn_edge_class(args, tr, va, te, w, C, seed):
    set_seed(seed)
    model = EdgeGCN(tr[0].x.size(1), args.hidden, args.embed, args.layers, args.dropout, num_edge_classes=C)
    wt = torch.from_numpy(w).to(DEVICE)
    ltr, lva, lte = loaders(args, tr, va, te)

    def logits(b):
        return model.classify_edges(model.encode(b.x, b.edge_index, b.edge_weight), b.edge_label_index)

    def train_epoch(opt):
        for b in ltr:
            b = b.to(DEVICE)
            if b.edge_label_class.numel() == 0:
                continue
            loss = F.cross_entropy(logits(b), b.edge_label_class, weight=wt)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def collect(loader):
        ys, ps = [], []
        for b in loader:
            b = b.to(DEVICE)
            ps.append(F.softmax(logits(b), 1))
            ys.append(b.edge_label_class)
        return torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()

    fit_torch(model, train_epoch, lambda: selection_score("multiclass", *collect(lva)), args.epochs,
              args.patience, args.lr, args.wd)
    (yv, pv), (yt, pt) = collect(lva), collect(lte)
    return score("multiclass", yv, pv, yt, pt), count_params(model)


def gcn_graph(args, tr, va, te, task, gidx, w, seed):
    set_seed(seed)
    cls = task == "graph_cls"
    kind = "binary" if cls else "regression"
    model = GraphGCN(tr[0].x.size(1), args.hidden, 2 if cls else 1, args.layers, args.dropout, args.pooling)
    wt = torch.from_numpy(w).to(DEVICE) if cls else None
    ltr, lva, lte = loaders(args, tr, va, te)

    def out_y(b):
        out = model(b.x, b.edge_index, b.batch, b.edge_weight)
        return out, (b.y_graph_cls if cls else b.y_graph[:, gidx])

    def train_epoch(opt):
        for b in ltr:
            out, y = out_y(b.to(DEVICE))
            loss = F.cross_entropy(out, y, weight=wt) if cls else F.mse_loss(out.squeeze(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def collect(loader):
        ys, ps = [], []
        for b in loader:
            out, y = out_y(b.to(DEVICE))
            ps.append(F.softmax(out, 1)[:, 1] if cls else out.squeeze(-1))
            ys.append(y)
        return torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()

    fit_torch(model, train_epoch, lambda: selection_score(kind, *collect(lva)), args.epochs,
              args.patience, args.lr, args.wd)
    (yv, pv), (yt, pt) = collect(lva), collect(lte)
    return score(kind, yv, pv, yt, pt), count_params(model)


# ------------------------------------- tabular runner -------------------------------------
def run_tabular(args, name, tables, kind, C, w, seed, res):
    (Xtr, ytr), (Xva, yva), (Xte, yte, *extra) = tables
    subset = extra[0] if extra else None
    ev = lambda pred: score(kind, yva, to_scores(kind, pred(Xva)), yte, to_scores(kind, pred(Xte)), subset)
    with Timer() as t:
        pred, n_par = fit_mlp(Xtr, ytr, Xva, yva, kind, C, args, w, seed)
    res.add(f"MLP{name}", ev(pred), n_par, t.seconds)
    with Timer() as t:
        pred = fit_gbm(Xtr, ytr, Xva, yva, kind, C, args, w, seed)
    res.add(f"GBM{name}", ev(pred), None, t.seconds)
    if kind != "ranking":
        res.add("Constant (prior/mean)", ev(constant_predictor(kind, ytr, C)), None, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/processed/gdelt.pt")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--out", default=None, help="default: results/gdelt_<task>.json")
    p.add_argument("--reg-target", choices=REG_TARGETS, default="conflict_share", help="node_reg target")
    p.add_argument("--graph-target", choices=REG_TARGETS, default="conflict_share", help="graph_reg target")
    p.add_argument("--embed", type=int, default=32, help="node embedding size for the edge-level GCN")
    p.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean", help="graph-level GCN readout")
    p.add_argument("--batch-size", type=int, default=8, help="snapshots per GCN batch")
    p.add_argument("--no-pair-history", action="store_true",
                   help="edge tasks: drop the pair's own lookback interaction features from MLP/GBM")
    add_model_args(p)
    p.set_defaults(lr=0.005)
    args = p.parse_args()
    args.out = args.out or f"results/gdelt_{args.task}.json"

    snaps, meta = load_processed(args.data)
    tr, va, te = snaps["train"], snaps["val"], snaps["test"]
    ridx, gidx = REG_TARGETS.index(args.reg_target), REG_TARGETS.index(args.graph_target)
    hist = not args.no_pair_history
    C = 4
    title = (f"GDELT {args.task} | {len(tr)}/{len(va)}/{len(te)} train/val/test snapshots, "
             f"{tr[0].num_nodes} countries, window {meta['freq']}")
    res = Results(title)

    # Fixed evaluation negatives so every model and seed is scored on the same pairs.
    if args.task == "link_prediction":
        for s, split in enumerate((va, te)):
            rng = np.random.default_rng(1234 + s)
            for d in split:
                add_negatives(d, rng)
        tr = [d for d in tr if d.edge_label_index.size(1) > 0]

    for seed in range(args.seeds):
        if args.task in ("node_class", "node_reg"):
            tabs = [node_table(s, args.task, ridx, False) for s in (tr, va, te)]
            w = balanced_class_weight(tabs[0][1], C) if args.task == "node_class" else None
            kind = "multiclass" if args.task == "node_class" else "regression"
            with Timer() as t:
                m, n_par = gcn_node(args, tr, va, te, args.task, ridx, w, C, seed)
            res.add("GCN", m, n_par, t.seconds)
            run_tabular(args, "", tabs, kind, C if kind == "multiclass" else 1, w, seed, res)
            if args.neighbor_baselines:
                tabs = [node_table(s, args.task, ridx, True) for s in (tr, va, te)]
                run_tabular(args, " + nbr-mean", tabs, kind, C if kind == "multiclass" else 1, w, seed, res)

        elif args.task == "link_prediction":
            with Timer() as t:
                m, n_par = gcn_link(args, tr, va, te, seed)
            res.add("GCN", m, n_par, t.seconds)
            rng = np.random.default_rng(seed)
            for d in tr:
                add_negatives(d, rng)
            tabs = [lp_table(s, hist) for s in (tr, va, te)]
            tabs = [(tabs[0][0], tabs[0][1]), (tabs[1][0], tabs[1][1]), tabs[2]]
            run_tabular(args, "", tabs, "ranking", 2, None, seed, res)

        elif args.task == "edge_class":
            tabs = [edge_class_table(s, hist) for s in (tr, va, te)]
            w = balanced_class_weight(tabs[0][1], C)
            with Timer() as t:
                m, n_par = gcn_edge_class(args, tr, va, te, w, C, seed)
            res.add("GCN", m, n_par, t.seconds)
            run_tabular(args, "", tabs, "multiclass", C, w, seed, res)

        else:                                           # graph_reg / graph_cls
            tabs = [graph_table(s, args.task, gidx) for s in (tr, va, te)]
            cls = args.task == "graph_cls"
            w = balanced_class_weight(tabs[0][1], 2) if cls else None
            with Timer() as t:
                m, n_par = gcn_graph(args, tr, va, te, args.task, gidx, w, seed)
            res.add("GCN", m, n_par, t.seconds)
            run_tabular(args, "", tabs, "binary" if cls else "regression", 2 if cls else 1, w, seed, res)
        print(f"seed {seed} done")

    print("\n" + res.table())
    res.save(args.out, args)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
