"""Why is the GCN worse than the MLP/GBM on Medicare? Trains GCN variants side by side.

  python diagnose_medicare_gcn.py --data data/processed/medicare.pt --seeds 3

Each row changes ONE thing from the baseline GCN, and reports AUPRC on the train / val / test providers
plus the epoch where validation peaked. How to read it:
  * train AUPRC high, val/test low  -> overfitting (try the less-regularised rows)
  * train AUPRC low too             -> underfitting: the model can't even fit the training providers
        - 'no edge weights' or '1 layer' better  -> neighbour averaging is washing out the node's own features
        - 'skip' better                          -> same cause, fixed by letting the node keep its own features
  * 'MLP (0 hops)' is the same network with no graph at all, the reference to beat.
"""
import argparse

import compare_utils  # noqa: F401  (must be first)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from compare_utils import (DEVICE, add_model_args, balanced_class_weight, fit_torch, note_loss, score,
                           selection_score, set_seed)
import compare_utils
from graph_utils import load_processed


class Net(torch.nn.Module):
    """GCN with `hops` message-passing layers (0 = plain MLP) and an optional skip of the raw features."""

    def __init__(self, in_ch, hidden, hops, dropout, skip):
        super().__init__()
        self.hops, self.dropout, self.skip = hops, dropout, skip
        dims = [in_ch] + [hidden] * max(hops, 1)
        self.convs = torch.nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(hops))
        self.pre = torch.nn.Linear(in_ch, hidden) if hops == 0 else None
        head_in = hidden + (in_ch if skip else 0)
        self.head = torch.nn.Sequential(torch.nn.Linear(head_in, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                        torch.nn.Linear(hidden, 2))

    def forward(self, x, ei, ew):
        h = x
        if self.hops == 0:
            h = F.relu(self.pre(h))
        for conv in self.convs:
            h = F.dropout(F.relu(conv(h, ei, ew)), self.dropout, self.training)
        if self.skip:
            h = torch.cat([h, x], 1)
        return self.head(h)


def run(name, data, w, seed, args, hops=2, use_w=True, skip=False, **over):
    a = argparse.Namespace(**{**vars(args), **over})
    set_seed(seed)
    d = data.to(DEVICE)
    model = Net(d.x.size(1), a.hidden, hops, a.dropout, skip)
    wt = torch.from_numpy(w).to(DEVICE)
    ew = d.edge_weight if use_w else None
    y = d.y.cpu().numpy()
    tr, va, te = (m.cpu().numpy() for m in (d.train_mask, d.val_mask, d.test_mask))

    def probs():
        return F.softmax(model(d.x, d.edge_index, ew), 1)[:, 1].cpu().numpy()

    def train_epoch(opt):
        opt.zero_grad()
        loss = F.cross_entropy(model(d.x, d.edge_index, ew)[d.train_mask], d.y[d.train_mask], weight=wt)
        loss.backward()
        note_loss(loss)
        opt.step()

    fit_torch(model, train_epoch, lambda: selection_score("binary", y[va], probs()[va]), a)
    fit = compare_utils.LAST_FIT
    compare_utils.LAST_FIT = None
    with torch.no_grad():
        p = probs()
    f = lambda m: score("binary", y[va], p[va], y[m], p[m])["auprc"]
    return {"train": f(tr), "val": f(va), "test": f(te), "best_epoch": fit["best_epoch"], "epochs": fit["epochs"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data/processed/medicare.pt")
    add_model_args(ap)
    args = ap.parse_args()
    data, _ = load_processed(args.data)
    y = data.y.numpy()
    w = balanced_class_weight(y[data.train_mask.numpy()], 2)
    ex = np.asarray(data.edge_weight) if data.edge_weight is not None else None
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
    print(f"graph: {data.num_nodes} nodes, {data.edge_index.size(1)} directed edges, mean degree {deg.mean():.1f}, "
          f"median {deg.median():.0f}, max {deg.max():.0f}, isolated nodes {(deg == 0).float().mean():.1%}")
    if ex is not None:
        print(f"edge weights: min {ex.min():.2f}, median {np.median(ex):.2f}, max {ex.max():.2f}")
    # homophily: fraction of edges joining providers with the same fraud label
    ei = data.edge_index
    print(f"edge homophily (same label): {(data.y[ei[0]] == data.y[ei[1]]).float().mean():.1%}  "
          f"(fraud rate {y.mean():.1%}; a graph only helps if linked providers share labels more than chance)\n")

    variants = [
        ("MLP (0 hops, reference)", dict(hops=0)),
        ("GCN baseline (2 hops, edge weights)", dict()),
        ("GCN no edge weights", dict(use_w=False)),
        ("GCN 1 hop", dict(hops=1)),
        ("GCN + skip (keeps own features)", dict(skip=True)),
        ("GCN + skip, 1 hop", dict(skip=True, hops=1)),
        ("GCN less regularised (drop 0.2, wd 1e-5)", dict(dropout=0.2, wd=1e-5)),
        ("GCN wider (128) + lower lr 0.003", dict(hidden=128, lr=0.003)),
    ]
    print(f"{'variant':44s} {'train':>7s} {'val':>7s} {'test':>7s}  best/epochs   (AUPRC, mean of {args.seeds} seeds)")
    for name, kw in variants:
        rs = [run(name, data.clone(), w, s, args, **kw) for s in range(args.seeds)]
        m = lambda k: np.nanmean([r[k] for r in rs])
        print(f"{name:44s} {m('train'):7.3f} {m('val'):7.3f} {m('test'):7.3f}  {m('best_epoch'):4.0f}/{m('epochs'):4.0f}",
              flush=True)


if __name__ == "__main__":
    main()
