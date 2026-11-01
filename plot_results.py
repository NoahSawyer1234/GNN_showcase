"""Plot the results/*.json files written by the compare_*.py scripts.

  python plot_results.py                          # reads results/*.json, writes figures/
  python plot_results.py --results results/medicare.json --theme dark --format svg

Per results file (one figure each): one panel per metric, one column per model.
  dot            one seed (jittered sideways so seeds don't overlap)
  thick bar      mean over seeds, with a thin +/- 1 std whisker
  colour         model family (graph models, GBM); a diamond marks "+ nbr-mean" (GBM given
                 hand-engineered neighbour-average features), a square "+ GAE" (GBM given unsupervised
                 graph-autoencoder embeddings)
  grey line      the trivial "predict the prior / mean" reference
The first panel is the headline metric.
Every panel has its own y-axis (one measure per axis, never a dual axis).

Also written: figures/overview.png (the headline metric of every task, graph models / GBM only) and
figures/summary.csv (the same numbers as a table, so nothing depends on reading a colour).
Colours are the validated 3-slot categorical palette (blue, orange, aqua) in both themes.
"""
import argparse
import csv
import glob
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", GCN="#2a78d6", GBM="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", GCN="#3987e5", GBM="#199e70"),
}
PRIMARY_PRIORITY = ["auprc", "macro_f1", "auroc", "mae"]
METRIC_ORDER = ["auprc", "macro_f1", "auroc", "ap", "f1", "accuracy", "auroc_new", "ap_new", "mae", "rmse", "r2"]
LOWER_BETTER = {"mae", "rmse"}
NAMES = {"auprc": "AUPRC", "auroc": "AUROC", "ap": "Average precision", "f1": "F1", "macro_f1": "Macro-F1",
         "accuracy": "Accuracy", "mae": "MAE", "rmse": "RMSE", "r2": "R²", "auroc_new": "AUROC, new links only",
         "ap_new": "AP, new links only"}
TASK_NAMES = {"medicare": "Medicare fraud · node classification",
              "underwriting_company": "Underwriting fraud · companies (out of time)",
              "underwriting_admin": "Underwriting fraud · administrators (out of time)",
              "wildfire_pixel": "Wildfire · pixel fire", "wildfire_patch": "Wildfire · patch burn fraction"}
FAMILIES = ("GCN", "GBM")
FAMILY_LABELS = {"GCN": "graph networks", "GBM": "GBM"}
MARKERS = {"nbr": "D", "GAE": "s"}            # GBM given graph information: hand-made / learned without labels
MARKER_LABELS = {"nbr": "+ neighbour-mean features", "GAE": "+ graph-autoencoder embeddings"}


# ------------------------------------------ loading ------------------------------------------
def load(path):
    d = json.loads(Path(path).read_text())
    models = {}
    for name, r in d["runs"].items():
        keys = [k for k in r["runs"][0] if k != "seconds"]
        models[name] = {k: np.array([run[k] for run in r["runs"]], dtype=float) for k in keys}
    return {"stem": Path(path).stem, "title": d.get("title", Path(path).stem), "models": models}


def is_reference(name):
    return name.startswith("Constant")


def family(name):
    """Colour group. Graph models (GCN, GCN + skip, GraphSAGE, GAT) share the GCN colour; every GBM
    row, including the GBM on graph-autoencoder embeddings, shares the GBM colour."""
    if name.startswith(("GraphSAGE", "GAT")):
        return "GCN"
    return next((f for f in FAMILIES if name.startswith(f)), None)


def metrics_of(res):
    present = {k for m in res["models"].values() for k, v in m.items() if np.isfinite(v).any()}
    ordered = [k for k in METRIC_ORDER if k in present] + sorted(present - set(METRIC_ORDER))
    head = next((k for k in PRIMARY_PRIORITY if k in ordered), ordered[0])
    return [head] + [k for k in ordered if k != head]


# ------------------------------------------ drawing ------------------------------------------
def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color=t["grid"], linewidth=0.7)            # solid hairline, recessive
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelsize=8, length=0, pad=4)


def label_for(name):
    return name.replace(" + nbr-mean", "\n+nbr").replace(" + skip", "\n+ skip").replace(" + GAE", "\n+GAE").replace("GraphSAGE", "Graph\nSAGE")


def draw_panel(ax, res, metric, models, t, rng, show_reference=True):
    style_axes(ax, t)
    xs = np.arange(len(models))
    all_vals = []
    for x, name in zip(xs, models):
        v = res["models"][name].get(metric, np.array([]))
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        all_vals.append(v)
        col = t.get(family(name), t["muted"])
        marker = MARKERS.get(next((k for k in MARKERS if k in name), None), "o")
        jitter = rng.uniform(-0.16, 0.16, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(x + jitter, v, s=46 if marker == "o" else 40, marker=marker, color=col, alpha=0.8,
                   edgecolors=t["surface"], linewidths=1.2, zorder=3)          # 2px surface ring
        mu, sd = v.mean(), v.std()
        if len(v) > 1:
            ax.vlines(x, mu - sd, mu + sd, color=col, linewidth=1.5, zorder=4)
        ax.hlines(mu, x - 0.3, x + 0.3, color=col, linewidth=2.6, zorder=5)
    refs = [n for n in res["models"] if is_reference(n)]
    if show_reference and refs:
        rv = res["models"][refs[0]].get(metric, np.array([]))
        rv = rv[np.isfinite(rv)]
        if len(rv):
            all_vals.append(rv)
            ax.axhline(rv.mean(), color=t["muted"], linewidth=1.0, zorder=2)
    # y-limits: padded data range; R² can be hugely negative on small data, so cap the view at -1
    lo = min(v.min() for v in all_vals)
    hi = max(v.max() for v in all_vals)
    clipped = False
    if metric == "r2" and lo < -1:
        lo, clipped = -1.0, True
    pad = (hi - lo) * 0.12 or 0.05
    ax.set_ylim(lo - pad, hi + pad)
    if clipped:
        ax.text(0.98, 0.02, "values below −1 not shown", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=t["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels([label_for(m) for m in models], fontsize=8, color=t["ink2"])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_title(f"{NAMES.get(metric, metric)}  ", loc="left", fontsize=10, color=t["ink"], fontweight="bold", pad=10)


def legend_handles(res, models, t):
    h = [Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=t[f], markeredgecolor=t["surface"],
                label=FAMILY_LABELS[f]) for f in FAMILIES if any(family(m) == f for m in models)]
    extras = [k for k in MARKERS if any(k in m for m in models)]
    if extras:
        h.append(Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=t["muted"],
                        markeredgecolor=t["surface"], label="own features"))
        h += [Line2D([0], [0], marker=MARKERS[k], linestyle="", markersize=5.5, markerfacecolor=t["muted"],
                     markeredgecolor=t["surface"], label=MARKER_LABELS[k]) for k in extras]
    if any(is_reference(n) for n in res["models"]):
        h.append(Line2D([0], [0], color=t["muted"], linewidth=1.0, label="predict prior / mean (reference)"))
    h.append(Line2D([0], [0], color=t["ink2"], linewidth=2.6, label="mean ± std over seeds (dots = seeds)"))
    return h


def task_figure(res, t, out_path, fmt, dpi):
    models = [m for m in res["models"] if not is_reference(m)]
    metrics = metrics_of(res)
    ncols = min(4, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    col_w = max(3.1, 0.55 * len(models))                      # room for every model's label
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_w * ncols + 0.6, 3.5 * nrows + 1.5),
                             squeeze=False, facecolor=t["surface"])
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes.ravel(), metrics):
        draw_panel(ax, res, metric, models, t, rng)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    title = "\n".join(textwrap.wrap(res["title"], 105))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=t["ink"], fontweight="bold")
    fig.legend(handles=legend_handles(res, models, t), loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1 if nrows == 1 else 0.06, 1, 0.93), h_pad=2.2, w_pad=1.6)
    fig.savefig(out_path, format=fmt, dpi=dpi, facecolor=t["surface"])
    plt.close(fig)


def overview_figure(results, t, out_path, fmt, dpi):
    ncols = 3 if len(results) > 4 else min(len(results), 2)
    nrows = int(np.ceil(len(results) / ncols))
    n_models = max(sum(not is_reference(m) and "nbr" not in m for m in r["models"]) for r in results)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(3.9, 0.6 * n_models) * ncols + 0.4, 3.3 * nrows + 1.3), squeeze=False,
                             facecolor=t["surface"])
    rng = np.random.default_rng(0)
    for ax, res in zip(axes.ravel(), results):
        models = [m for m in res["models"] if not is_reference(m) and "nbr" not in m]
        metric = metrics_of(res)[0]
        draw_panel(ax, res, metric, models, t, rng)
        task = TASK_NAMES.get(res["stem"], res["stem"])
        ax.set_title(f"{task}  ·  {NAMES.get(metric, metric)}", loc="left", fontsize=9, color=t["ink"],
                     fontweight="bold", pad=10)
    for ax in axes.ravel()[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Graph models vs GBM: headline metric per task (test set)", x=0.01, ha="left", fontsize=11,
                 color=t["ink"], fontweight="bold")
    fig.legend(handles=legend_handles(results[0], FAMILIES, t)[:len(FAMILIES)] +
               [Line2D([0], [0], color=t["muted"], linewidth=1.0, label="predict prior / mean (reference)"),
                Line2D([0], [0], color=t["ink2"], linewidth=2.6, label="mean ± std over seeds (dots = seeds)")],
               loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=t["ink2"],
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=2.2, w_pad=1.6)
    fig.savefig(out_path, format=fmt, dpi=dpi, facecolor=t["surface"])
    plt.close(fig)


def write_table(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "metric", "mean", "std", "n_seeds"])
        for res in results:
            for model, ms in res["models"].items():
                for k, v in ms.items():
                    v = v[np.isfinite(v)]
                    if len(v):
                        w.writerow([res["stem"], model, k, f"{v.mean():.6f}", f"{v.std():.6f}", len(v)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", nargs="+", default=["results/*.json"], help="files or globs")
    p.add_argument("--out", default="figures")
    p.add_argument("--theme", choices=list(THEMES), default="light")
    p.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    files = sorted({f for pat in args.results for f in glob.glob(pat)})
    if not files:
        raise FileNotFoundError(f"No result files match {args.results}. Run a compare_*.py script first.")
    t = THEMES[args.theme]
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Segoe UI", "Arial", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [load(f) for f in files]
    for res in results:
        path = out / f"{res['stem']}.{args.format}"
        task_figure(res, t, path, args.format, args.dpi)
        print(f"wrote {path}")
    if len(results) > 1:
        path = out / f"overview.{args.format}"
        overview_figure(results, t, path, args.format, args.dpi)
        print(f"wrote {path}")
    write_table(results, out / "summary.csv")
    print(f"wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
