"""Why does the GCN trail the GBM on Medicare?  Graph facts + GCN variants side by side.

  python diagnose_medicare_gcn.py --data data/processed/medicare.pt --seeds 3

Part 1, graph facts (no training)
  * degree distribution overall and by class, and how well degree alone ranks fraud (AUPRC)
  * edge-weight range, and the share of its own row a node keeps in GCN normalisation (self-loop share)
  * label homophily: edge homophily, class-adjusted homophily (0 = no better than random wiring),
    mean fraud share of neighbours for fraud vs non-fraud providers, and the AUPRC of
    "fraud share of my TRAINING-labelled neighbours" on the test providers (label propagation in one hop)
Part 2, linear probes: logistic regression on X, ÂX, Â²X and [X, ÂX] (Â = GCN-normalised adjacency).
  If ÂX scores far below X, the neighbour averaging is destroying each provider's own signal.
Part 3, GCN variants with train / val / test AUPRC (same training loop, early stopping on validation):
  1 / 2 hops, with / without edge weights, skip connection, GraphSAGE, less regularisation;
  the GBM on the provider's own features is the reference row.
A large train-test gap means overfitting; a low TRAIN score means the model cannot fit (underfitting).
"""
import argparse

import compare_utils  # noqa: F401  (must be first)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from compare_utils import (DEVICE, add_model_args, balanced_class_weight, fit_gbm, fit_torch, note_loss,
                           set_seed)
from graph_utils import load_processed
from models import NodeGCN


def ap(y, p):
    return float(average_precision_score(y, p))


# ------------------------------------ part 1 ------------------------------------
def graph_facts(d):
    n, (src, dst), w, y = d.num_nodes, d.edge_index, d.edge_weight, d.y
    tr, te = d.train_mask, d.test_mask
    deg = torch.bincount(dst, minlength=n).float()
    pct = lambda t: "  ".join(f"p{q}={np.percentile(t.numpy(), q):.1f}" for q in (0, 10, 50, 90, 99, 100))
    print("\n=== 1. graph facts ===")
    print(f"nodes {n}, directed edges {src.numel()}, isolated {(deg == 0).sum().item()}, "
          f"fraud rate {y.float().mean():.3f}")
    print(f"degree:       {pct(deg)}")
    print(f"  median degree  fraud {deg[y == 1].median():.0f}   non-fraud {deg[y == 0].median():.0f}")
    print(f"  AUPRC of degree alone on test: {ap(y[te], deg[te]):.3f}   (prior {y[te].float().mean():.3f})")
    print(f"edge weight:  {pct(w)}   (max/min ratio {w.max() / w.min():.1f})")

    # Share of its own (normalised) row a node keeps after gcn_norm adds a self-loop of weight 1.
    ei, nw = gcn_norm(d.edge_index, w, n, add_self_loops=True)
    self_mask = ei[0] == ei[1]
    row_sum = scatter(nw, ei[1], dim=0, dim_size=n, reduce="sum")
    self_share = torch.zeros(n).index_add_(0, ei[1][self_mask], nw[self_mask]) / row_sum
    print(f"self share of GCN row (weighted): median {self_share.median():.3f}, "
          f"fraud {self_share[y == 1].median():.3f}, non-fraud {self_share[y == 0].median():.3f}")

    same = (y[src] == y[dst]).float().mean().item()
    # Adjusted homophily (Platonov et al. 2023): edge homophily minus what degree-weighted random wiring gives.
    dk = torch.stack([deg[y == c].sum() for c in (0, 1)]) / deg.sum()
    adj = (same - (dk ** 2).sum().item()) / (1 - (dk ** 2).sum().item())
    nb_frac = torch.zeros(n).index_add_(0, dst, y[src].float()) / deg.clamp(min=1)
    print(f"edge homophily {same:.3f}   class-adjusted homophily {adj:+.3f}  (0 = random wiring, 1 = perfect)")
    print(f"mean fraud share of neighbours:  fraud providers {nb_frac[y == 1].mean():.3f}   "
          f"non-fraud {nb_frac[y == 0].mean():.3f}")
    # One-hop label propagation using ONLY training labels: does knowing my neighbours' labels help?
    lab = (y.float() * tr.float())[src]
    known = tr.float()[src]
    num = torch.zeros(n).index_add_(0, dst, lab)
    den = torch.zeros(n).index_add_(0, dst, known)
    prop = torch.where(den > 0, num / den.clamp(min=1), y[tr].float().mean())
    print(f"AUPRC on test of 'fraud share of training-labelled neighbours': {ap(y[te], prop[te]):.3f}")
    return deg


# ------------------------------------ part 2 ------------------------------------
def propagate(x, edge_index, edge_weight, n, hops):
    ei, nw = gcn_norm(edge_index, edge_weight, n, add_self_loops=True)
    for _ in range(hops):
        x = scatter(x[ei[0]] * nw[:, None], ei[1], dim=0, dim_size=n, reduce="sum")
    return x


def linear_probe(x, d, w):
    """Logistic regression (full batch, early stopped on validation AUPRC). Returns train/val/test AUPRC."""
    set_seed(0)
    x = (x - x[d.train_mask].mean(0)) / x[d.train_mask].std(0).clamp(min=1e-6)
    lin = torch.nn.Linear(x.size(1), 2)
    opt = torch.optim.Adam(lin.parameters(), lr=0.01)
    wt = torch.from_numpy(w)
    best, best_state, bad = -1, None, 0
    for _ in range(3000):
        opt.zero_grad()
        F.cross_entropy(lin(x[d.train_mask]), d.y[d.train_mask], weight=wt).backward()
        opt.step()
        with torch.no_grad():
            s = ap(d.y[d.val_mask], F.softmax(lin(x[d.val_mask]), 1)[:, 1])
        if s > best + 1e-4:
            best, bad, best_state = s, 0, {k: v.clone() for k, v in lin.state_dict().items()}
        else:
            bad += 1
            if bad >= 100:
                break
    lin.load_state_dict(best_state)
    with torch.no_grad():
        p = F.softmax(lin(x), 1)[:, 1]
    return [ap(d.y[m], p[m]) for m in (d.train_mask, d.val_mask, d.test_mask)]


def probes(d, w):
    print("\n=== 2. linear probes (logistic regression; AUPRC train / val / test) ===")
    n, x = d.num_nodes, d.x
    ax = propagate(x, d.edge_index, d.edge_weight, n, 1)
    a2x = propagate(x, d.edge_index, d.edge_weight, n, 2)
    ax_u = propagate(x, d.edge_index, None, n, 1)
    for name, f in [("X (own features)", x), ("ÂX (1 hop, weighted)", ax), ("ÂX (1 hop, unweighted)", ax_u),
                    ("Â²X (2 hops)", a2x), ("[X, ÂX]", torch.cat([x, ax], 1))]:
        r = linear_probe(f, d, w)
        print(f"  {name:<26} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}", flush=True)


# ------------------------------------ part 3 ------------------------------------
VARIANTS = [  # name, hops, conv, use edge weight, dropout, weight decay (None = CLI value)
    ("GCN 1 layer", 1, "gcn", True, None, None),
    ("GCN 2 layers (plain)", 2, "gcn", True, None, None),
    ("GCN 2L, no edge weight", 2, "gcn", False, None, None),
    ("GCN 2L, less reg", 2, "gcn", True, 0.1, 0.0),
    ("GCN 2L + skip", 2, "gcn_skip", True, None, None),
    ("GCN 2L + skip, no edge weight", 2, "gcn_skip", False, None, None),
    ("GraphSAGE 2L", 2, "sage", False, None, None),
]


def run_variant(d, w, args, seed, hops, conv, use_w, dropout, wd):
    set_seed(seed)
    a = argparse.Namespace(**vars(args))
    a.dropout = args.dropout if dropout is None else dropout
    a.wd = args.wd if wd is None else wd
    model = NodeGCN(d.x.size(1), a.hidden, 2, hops, a.dropout, conv=conv)
    ew = d.edge_weight if use_w else None
    forward = lambda: model(d.x, d.edge_index, ew)
    wt = torch.from_numpy(w).to(DEVICE)
    y = d.y.cpu().numpy()
    masks = [m.cpu().numpy() for m in (d.train_mask, d.val_mask, d.test_mask)]

    def probs():
        return F.softmax(forward(), 1)[:, 1].detach().cpu().numpy()

    def train_epoch(opt):
        opt.zero_grad()
        loss = F.cross_entropy(forward()[d.train_mask], d.y[d.train_mask], weight=wt)
        loss.backward()
        note_loss(loss)
        opt.step()

    model.to(DEVICE)
    fit_torch(model, train_epoch, lambda: ap(y[masks[1]], probs()[masks[1]]), a)
    fit = compare_utils.LAST_FIT
    compare_utils.LAST_FIT = None
    with torch.no_grad():
        p = probs()
    return [ap(y[m], p[m]) for m in masks], fit["best_epoch"]


def gbm_reference(d, w, args):
    """GBM on the provider's own features (the comparison baseline): train / val / test AUPRC."""
    x, y = d.x.numpy(), d.y.numpy()
    tr, va = d.train_mask.numpy(), d.val_mask.numpy()
    pred = fit_gbm(x[tr], y[tr], x[va], y[va], "binary", 2, args, w, 0)
    p = pred(x)[:, 1]
    return [ap(y[m], p[m]) for m in (tr, va, d.test_mask.numpy())]


def variants(d, w, args):
    print(f"\n=== 3. variants (AUPRC mean over {args.seeds} seeds; hidden {args.hidden}, "
          f"dropout {args.dropout}, wd {args.wd} unless stated) ===")
    print(f"  {'variant':<32}{'train':>8}{'val':>8}{'test':>8}{'test sd':>9}{'best ep':>9}")
    g = gbm_reference(d, w, args)
    print(f"  {'GBM, own features (reference)':<32}{g[0]:>8.3f}{g[1]:>8.3f}{g[2]:>8.3f}{'-':>9}{'-':>9}", flush=True)
    dd = d.to(DEVICE)
    for name, hops, conv, use_w, dr, wd in VARIANTS:
        rs, eps = [], []
        for seed in range(args.seeds):
            r, ep = run_variant(dd, w, args, seed, hops, conv, use_w, dr, wd)
            rs.append(r)
            eps.append(ep)
        rs = np.array(rs)
        mu = rs.mean(0)
        print(f"  {name:<32}{mu[0]:>8.3f}{mu[1]:>8.3f}{mu[2]:>8.3f}{rs[:, 2].std():>9.3f}"
              f"{np.mean(eps):>9.0f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/processed/medicare.pt")
    add_model_args(p)
    args = p.parse_args()
    d, _ = load_processed(args.data)
    w = balanced_class_weight(d.y[d.train_mask].numpy(), 2)
    graph_facts(d)
    probes(d, w)
    variants(d, w, args)


if __name__ == "__main__":
    main()
