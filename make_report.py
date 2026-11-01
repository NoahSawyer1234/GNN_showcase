"""Interactive HTML summary of the results/*.json files written by the compare_*.py scripts.

  python make_report.py                      # reads results/*.json, writes figures/report.html
  python make_report.py --results results/medicare.json --out /tmp/medicare_report.html

One self-contained file (no internet needed, no external libraries), for presenting the comparison:
  at a glance    per task: the best graph model vs the GBM on the headline metric, in words and numbers
  overview       the headline metric of every task, one panel each (dots = seeds, bar = mean, whisker = std)
  explore        pick a task and a metric; the same chart, larger, plus a plain-language note on the metric
  table          every model x every metric as mean +- std, so nothing depends on reading a colour
Controls: show / hide the "GBM + neighbour features" rows and the constant-guess reference line; light /
dark theme. Hover or focus a column for the per-seed values, parameter count and training epochs.
"Level" in the at-a-glance cards means the gap is under max(0.01, 2 x the graph model's seed std) on a
0-1 score (5% of the GBM's error for error metrics). Seed std covers training randomness only, not
test-set sampling noise, so small gaps should not be over-read.
"""
import argparse
import glob
import json
from datetime import date
from pathlib import Path

import numpy as np

from plot_results import LOWER_BETTER, NAMES, TASK_NAMES, family, is_reference, load, metrics_of

# What each task asks, in words for a non-specialist audience.
SETUP = {
    "medicare": ("Is a healthcare provider under investigation for Medicare fraud?",
                 "Providers are linked when they share patients or physicians. Random 60/20/20 split of providers."),
    "underwriting_company": ("Will a company be confirmed as committing underwriting fraud?",
                             "Belgian insurer data: companies are linked to their directors and to each other. Models "
                             "learn on an earlier snapshot of the network and are tested on companies first labelled "
                             "in a later one."),
    "underwriting_admin": ("Will a company director be confirmed as involved in underwriting fraud?",
                           "Same network as above, predicting the directors. A director's own data is only age and "
                           "gender, so most information sits in the companies they run."),
    "wildfire_pixel": ("Will this 1 km square be burning tomorrow?",
                       "Each square is linked to its 8 neighbours. Inputs: weather, terrain, vegetation, population "
                       "and today's fire."),
    "wildfire_patch": ("What share of a 64 x 64 km area will be burning tomorrow?",
                       "One graph per area (its 4,096 squares); the graph models summarise the whole area."),
}
METRIC_HELP = {
    "auprc": "Area under the precision-recall curve. How well a model ranks the rare positive cases (fraud, fire) "
             "above everything else. 1 = perfect; guessing scores about the share of positives (grey line).",
    "auroc": "The chance that a randomly chosen positive case is ranked above a randomly chosen negative one. "
             "0.5 = coin flip, 1 = perfect.",
    "f1": "Balance between catching positives (recall) and not raising false alarms (precision), at the decision "
          "threshold picked on the validation set. 1 = perfect.",
    "accuracy": "Share of cases classified correctly at that threshold. Can look high when positives are rare: "
                "always answering 'no' already scores the share of negatives.",
    "mae": "Mean absolute error: the average size of a miss, in the target's own units. 0 = perfect.",
    "rmse": "Root mean squared error: like the mean absolute error, but large misses count extra. 0 = perfect.",
    "r2": "Share of the variation in the target that the model explains. 1 = perfect, 0 = no better than always "
          "predicting the average, below 0 = worse than that.",
}
DISPLAY = {"GBM + nbr-mean": "GBM + neighbour features", "GBM + GAE": "GBM + GAE embeddings"}
GRAPH_GBMS = {"GBM + nbr-mean": "GBM + neighbours", "GBM + GAE": "GBM + GAE"}   # GBM given graph information
MODEL_HELP = {
    "GCN": "Graph convolutional network: each node averages itself with its neighbours, layer by layer.",
    "GCN + skip": "GCN with an extra path that keeps each node's own features, so they are not averaged away.",
    "GraphSAGE": "Graph network that combines a node's own features with the mean of its neighbours' features.",
    "GAT": "Graph attention network: each node learns how much weight to give each of its neighbours.",
    "GBM + GAE": "Gradient-boosted trees given each node's own features plus an embedding from a graph "
                 "autoencoder: a graph network trained, without any labels, to reconstruct which nodes are linked.",
    "GBM": "Gradient-boosted trees on each node's own features only (no graph).",
    "GBM + nbr-mean": "Gradient-boosted trees given each node's own features plus the average of its "
                      "neighbours' features: hand-made graph information.",
}


def verdict(task):
    """Best graph model vs plain GBM on the headline metric, in words."""
    m = task["primary"]
    lower = m in LOWER_BETTER
    stats = {md["name"]: md["stats"][m] for md in task["models"] if m in md["stats"]}
    graph = [n for n in stats if family(n) == "GCN"]
    if not graph or "GBM" not in stats:
        return None
    best = (min if lower else max)(graph, key=lambda n: stats[n]["mean"])
    g, b = stats[best], stats["GBM"]
    gain = (b["mean"] - g["mean"]) if lower else (g["mean"] - b["mean"])
    tol = max(0.05 * abs(b["mean"]), 2 * g["sd"]) if lower else max(0.01, 2 * g["sd"])
    word = "about level with" if abs(gain) < tol else ("ahead of" if gain > 0 else "behind")
    out = {"best": best, "best_mean": g["mean"], "best_sd": g["sd"], "gbm_mean": b["mean"], "gain": gain,
           "word": word}
    extra = [n for n in GRAPH_GBMS if n in stats]
    if extra:                                   # the better GBM that was given graph information
        n = (min if lower else max)(extra, key=lambda k: stats[k]["mean"])
        out["graph_gbm"], out["graph_gbm_mean"] = GRAPH_GBMS[n], stats[n]["mean"]
    return out


def task_payload(path):
    res = load(path)
    raw = json.loads(Path(path).read_text())
    metrics = metrics_of(res)
    models = []
    for name, vals in res["models"].items():
        r = raw["runs"][name]
        stats = {}
        for k in metrics:
            v = vals.get(k, np.array([]))
            v = v[np.isfinite(v)]
            if len(v):
                stats[k] = {"mean": float(v.mean()), "sd": float(v.std()), "seeds": [float(x) for x in v]}
        fit = r.get("fit")
        models.append({
            "name": name, "label": DISPLAY.get(name, name), "family": family(name) or "ref",
            "nbr": name in GRAPH_GBMS, "marker": "diamond" if "nbr" in name else "square" if "GAE" in name else "circle",
            "reference": is_reference(name), "params": r.get("params"),
            "seconds": float(np.mean([x.get("seconds") or 0 for x in r["runs"]])),
            "epochs": float(np.mean([f["epochs"] for f in fit])) if fit else None,
            "best_epoch": float(np.mean([f["best_epoch"] for f in fit])) if fit else None,
            "help": MODEL_HELP.get(name, "Reference: always predicts the training average / share of positives."
                                   if is_reference(name) else ""),
            "stats": stats})
    stem = res["stem"]
    args = raw.get("args", {})
    task = {"stem": stem, "name": TASK_NAMES.get(stem, stem), "title": raw.get("title", stem),
            "question": SETUP.get(stem, ("", ""))[0], "setup": SETUP.get(stem, ("", ""))[1],
            "primary": metrics[0], "metrics": metrics, "metric_names": {k: NAMES.get(k, k) for k in metrics},
            "lower": [k for k in metrics if k in LOWER_BETTER],
            "n_seeds": max(len(s["seeds"]) for md in models for s in md["stats"].values()),
            "settings": {k: args.get(k) for k in ("hidden", "layers", "dropout", "lr", "wd", "max_train_samples")
                         if k in args},
            "models": models}
    task["verdict"] = verdict(task)
    return task


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", nargs="+", default=["results/*.json"], help="files or globs")
    p.add_argument("--out", default="figures/report.html")
    args = p.parse_args()
    files = sorted({f for pat in args.results for f in glob.glob(pat)})
    if not files:
        raise FileNotFoundError(f"No result files match {args.results}. Run a compare_*.py script first.")
    order = list(SETUP)
    tasks = sorted((task_payload(f) for f in files),
                   key=lambda t: order.index(t["stem"]) if t["stem"] in order else len(order))
    data = {"generated": date.today().isoformat(), "tasks": tasks, "metric_help": METRIC_HELP}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {out} ({len(tasks)} task{'s' if len(tasks) != 1 else ''}: {', '.join(t['stem'] for t in tasks)})")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Graphs vs Boosting</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.045);
  --graph: #2a78d6; --gbm: #1baf7a; --ref: #898781; --accent-ink: #1c5cab;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.06);
    --graph: #3987e5; --gbm: #199e70; --ref: #898781; --accent-ink: #86b6ef;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.06);
  --graph: #3987e5; --gbm: #199e70; --ref: #898781; --accent-ink: #86b6ef;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 28px 16px 64px; }
header.top { display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; }
h1 { font-size: 26px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 18px; margin: 40px 0 4px; }
h3 { font-size: 15px; margin: 0 0 2px; }
p { margin: 0 0 10px; }
.sub { color: var(--ink2); max-width: 70ch; }
.small { font-size: 13px; color: var(--ink2); }
.muted { color: var(--muted); }
button.theme { flex: none; font: inherit; font-size: 13px; color: var(--ink2); background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; cursor: pointer; }
button.theme:hover { background: var(--wash); }
.filters { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; margin: 22px 0 6px;
  padding: 10px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
.filters label { display: inline-flex; gap: 6px; align-items: center; font-size: 14px; color: var(--ink2); cursor: pointer; }
.filters select { font: inherit; font-size: 14px; color: var(--ink); background: var(--page); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink2); margin: 8px 0 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  text-align: left; font: inherit; color: inherit; cursor: pointer; }
.card:hover, .card:focus-visible { background: color-mix(in srgb, var(--surface), var(--ink) 4%); outline: none; }
.card .task { font-size: 13px; color: var(--ink2); font-weight: 600; }
.card .q { font-size: 13px; color: var(--ink2); margin: 2px 0 10px; min-height: 2.6em; }
.card .nums { display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }
.card .big { font-size: 26px; font-weight: 650; line-height: 1.1; }
.card .lab { font-size: 12px; color: var(--ink2); }
.card .verdict { margin-top: 10px; font-size: 14px; }
.card .verdict b { font-weight: 650; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); gap: 12px; margin-top: 12px; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 14px 8px; min-width: 0; }
.panel .head { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.panel .head button { font: inherit; font-size: 12px; color: var(--accent-ink); background: none; border: 0; padding: 0; cursor: pointer; white-space: nowrap; }
.chart { width: 100%; }
.chart svg { display: block; max-width: 100%; overflow: visible; }
.explore { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-top: 12px; }
.explore .controls { display: flex; flex-wrap: wrap; gap: 10px 18px; margin-bottom: 8px; }
.explore .controls label { font-size: 14px; color: var(--ink2); display: inline-flex; gap: 6px; align-items: center; max-width: 100%; min-width: 0; }
.explore select { font: inherit; font-size: 14px; color: var(--ink); background: var(--page); border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px; min-width: 0; max-width: 100%; flex: 1 1 auto; }
.help { font-size: 14px; color: var(--ink2); background: var(--page); border-radius: 8px; padding: 10px 12px; margin-top: 8px; }
.tablewrap { overflow-x: auto; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
th.sel { color: var(--ink); box-shadow: inset 0 -2px 0 var(--ink); }
td.best { font-weight: 700; }
td .sd { color: var(--muted); font-weight: 400; }
.notes { font-size: 14px; color: var(--ink2); }
.notes li { margin-bottom: 6px; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.xl { fill: var(--ink2); font-size: 12px; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.baseline { stroke: var(--axis); stroke-width: 1; }
.reflin { stroke: var(--ref); stroke-width: 1.25; stroke-dasharray: 4 3; }
.reflab { fill: var(--muted); font-size: 11px; }
.c-GCN { fill: var(--graph); stroke: var(--graph); }
.c-GBM { fill: var(--gbm); stroke: var(--gbm); }
.c-ref { fill: var(--ref); stroke: var(--ref); }
.dot { stroke: var(--surface); stroke-width: 2; fill-opacity: 0.85; }
.meanbar { stroke-width: 3; stroke-linecap: round; fill: none; }
.whisk { stroke-width: 1.5; fill: none; }
.hit { fill: transparent; cursor: default; outline: none; }
.col.on .hit, .hit:focus-visible { fill: var(--wash); }
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.45;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); max-width: 280px; opacity: 0; transition: opacity 0.08s; }
.tip .v { font-size: 15px; font-weight: 700; font-variant-numeric: tabular-nums; }
.tip .n { color: var(--ink2); display: flex; align-items: center; gap: 6px; }
.tip .n i { display: inline-block; width: 12px; height: 2px; }
.tip .d { color: var(--ink2); margin-top: 4px; }
footer { margin-top: 40px; font-size: 12px; color: var(--muted); }
@media (max-width: 560px) { h1 { font-size: 22px; } .card .big { font-size: 22px; } }
</style>
</head>
<body>
<main>
  <header class="top">
    <div>
      <h1>Do graph neural networks beat gradient boosting?</h1>
      <p class="sub">Graph neural networks (GCN, GCN&nbsp;+&nbsp;skip, GraphSAGE, GAT) against gradient-boosted trees
        (GBM) on the same data and the same train / validation / test split. Every model was tuned and stopped on
        validation data only; all numbers below are on the held-out test set.</p>
    </div>
    <button class="theme" id="theme" type="button">Dark theme</button>
  </header>

  <div class="filters" role="group" aria-label="Filters for every chart below">
    <label><input type="checkbox" id="f-nbr" checked> Show GBM given graph information</label>
    <label><input type="checkbox" id="f-ref" checked> Show constant-guess reference</label>
  </div>
  <div class="legend" aria-hidden="true">
    <span><i class="swatch" style="background:var(--graph)"></i>Graph models</span>
    <span><i class="swatch" style="background:var(--gbm)"></i>GBM (no graph)</span>
    <span><svg width="12" height="12"><rect x="2.5" y="2.5" width="7" height="7" transform="rotate(45 6 6)" fill="var(--gbm)"/></svg>GBM + neighbour features</span>
    <span><svg width="12" height="12"><rect x="2" y="2" width="8" height="8" rx="1" fill="var(--gbm)"/></svg>GBM + graph-autoencoder embeddings</span>
    <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" class="reflin"/></svg>Constant guess</span>
    <span class="muted">Dots: one per training seed · bar: mean · whisker: ±1 std</span>
  </div>

  <h2>At a glance</h2>
  <p class="small">Best graph model against the GBM on each task's headline metric. Click a card to explore it.</p>
  <div class="cards" id="cards"></div>

  <h2>Headline metric, every task</h2>
  <div class="panels" id="panels"></div>

  <h2 id="explore-h">Explore a task</h2>
  <section class="explore" aria-labelledby="explore-h">
    <div class="controls">
      <label>Task <select id="sel-task"></select></label>
      <label>Metric <select id="sel-metric"></select></label>
    </div>
    <h3 id="ex-q"></h3>
    <p class="small" id="ex-setup"></p>
    <div class="chart" id="ex-chart"></div>
    <div class="help" id="ex-help"></div>
    <div class="tablewrap"><table id="ex-table"></table></div>
    <p class="small" id="ex-meta" style="margin-top:10px"></p>
  </section>

  <h2>How to read this</h2>
  <ul class="notes">
    <li><b>Same data for every model.</b> The graph networks also see the links between nodes; the plain GBM sees only
      each node's own features. Two GBM rows are also given graph information: "GBM + neighbour features" gets the
      average of each node's neighbours as extra columns, and "GBM + GAE" gets embeddings from a graph autoencoder,
      a graph network trained without labels to reconstruct the links. Together they separate "does the graph
      help?" from "does a graph <i>network</i> help?".</li>
    <li><b>Wildfire</b> has no autoencoder row: every area has the same grid of links, so there is no structure to learn.</li>
    <li><b>Spread.</b> Dots are repeated training runs with different random seeds. The GBM is deterministic here, so its
      dots coincide. The spread does not include test-set sampling noise, so small gaps between models
      should not be over-read; "about level" in the cards marks gaps under 0.01 (or twice the seed spread).</li>
    <li><b>Medicare.</b> Linked providers are no more likely to share a fraud label than random pairs, so the graph adds
      little; the plain GCN also averages away each provider's own features.</li>
    <li><b>Underwriting data</b>: Vandervorst, Deprez, Verbeke &amp; Verdonck (2026), <i>Data Mining and Knowledge
      Discovery</i> 40(5):78; Allianz Benelux, CC BY-NC 4.0. <b>Wildfire data</b>: Huot et al., Next Day Wildfire Spread.
      <b>Medicare data</b>: Kaggle, healthcare-provider-fraud-detection-analysis.</li>
  </ul>
  <footer id="foot"></footer>
</main>
<div class="tip" id="tip" role="tooltip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const TASKS = DATA.tasks;
  const state = { task: TASKS.length ? TASKS[0].stem : null, metric: null, nbr: true, ref: true };
  const SVGNS = "http://www.w3.org/2000/svg";
  const byStem = Object.fromEntries(TASKS.map(t => [t.stem, t]));

  // ------------------------------------------------------------ helpers
  function el(tag, attrs, text) {
    const e = tag.startsWith("svg:") ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function fmt(v) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    const a = Math.abs(v);
    return a >= 0.1 || a === 0 ? v.toFixed(3) : v.toPrecision(3);
  }
  function fmtSd(sd, mean) { return Math.abs(mean) >= 0.1 ? sd.toFixed(3) : sd.toPrecision(2); }
  function spread(s) { return s.seeds.length > 1 && !s.seeds.every(v => v === s.seeds[0]); }
  function fmtGain(v, ref) {  // same decimals as the values it compares
    const a = Math.abs(v), s = Math.abs(ref) >= 0.1 ? a.toFixed(3) : a.toPrecision(2);
    return (v >= 0 ? "+" : "−") + s;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    const step = (r >= 7.5 ? 10 : r >= 3.5 ? 5 : r >= 1.5 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function visibleModels(task) {
    return task.models.filter(m => !m.reference && (state.nbr || !m.nbr));
  }
  function wrapLabel(label, narrow) {
    if (label === "GBM + neighbour features") return ["GBM +", "neighbours"];
    if (label === "GBM + GAE embeddings") return ["GBM +", "GAE"];
    if (narrow && label === "GraphSAGE") return ["Graph", "SAGE"];
    const i = label.indexOf(" + ");
    return i < 0 ? [label] : [label.slice(0, i), label.slice(i + 1)];
  }

  // ------------------------------------------------------------ tooltip
  const tip = document.getElementById("tip");
  function showTip(model, metricKey, task, x, y) {
    const s = model.stats[metricKey];
    tip.replaceChildren();
    tip.append(el("div", { class: "v" }, s ? fmt(s.mean) + (spread(s) ? "  ± " + fmtSd(s.sd, s.mean) : "") : "–"));
    const n = el("div", { class: "n" });
    const key = el("i"); key.style.background = model.family === "GCN" ? "var(--graph)" : model.family === "GBM" ? "var(--gbm)" : "var(--ref)";
    n.append(key, document.createTextNode(model.label + " · " + task.metric_names[metricKey]));
    tip.append(n);
    if (s && s.seeds.length > 1) tip.append(el("div", { class: "d" }, s.seeds.every(v => v === s.seeds[0])
      ? `Identical on all ${s.seeds.length} seeds` : "Seeds: " + s.seeds.map(fmt).join(", ")));
    const bits = [];
    if (model.params) bits.push(model.params.toLocaleString() + " parameters");
    if (model.epochs) bits.push("trained " + Math.round(model.epochs) + " epochs, best at " + Math.round(model.best_epoch));
    if (bits.length) tip.append(el("div", { class: "d" }, bits.join(" · ")));
    if (model.help) tip.append(el("div", { class: "d" }, model.help));
    tip.style.opacity = "1";
    placeTip(x, y);
  }
  function placeTip(x, y) {
    const r = tip.getBoundingClientRect(), pad = 14;
    let left = x + pad, top = y + pad;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { tip.style.opacity = "0"; }

  // ------------------------------------------------------------ dot plot
  // Charts redraw whenever their container changes width, so the SVG always matches the layout.
  const ro = new ResizeObserver(entries => {
    for (const e of entries) {
      const c = e.target;
      if (c._draw && Math.abs(c.clientWidth - (c._w || 0)) > 1) c._draw();
    }
  });
  function dotPlot(container, task, metricKey, height) {
    container._draw = () => drawDots(container, task, metricKey, height);
    ro.observe(container);
    container._draw();
  }
  function drawDots(container, task, metricKey, height) {
    container.replaceChildren();
    container._w = container.clientWidth;
    const width = Math.max(260, container.clientWidth || 320);
    const models = visibleModels(task).filter(m => m.stats[metricKey]);
    const ref = state.ref ? task.models.find(m => m.reference && m.stats[metricKey]) : null;
    const narrow = width < 520;                     // phones: shorter labels, reference named by the legend only
    const M = { l: 46, r: ref && !narrow ? 56 : 12, t: 12, b: 40 };
    const svg = el("svg:svg", { viewBox: `0 0 ${width} ${height}`, width: width, height: height, role: "img",
      "aria-label": `${task.name}: ${task.metric_names[metricKey]} by model` });
    if (!models.length) { container.append(svg); return; }
    let vals = models.flatMap(m => m.stats[metricKey].seeds.concat([m.stats[metricKey].mean - m.stats[metricKey].sd, m.stats[metricKey].mean + m.stats[metricKey].sd]));
    if (ref) vals.push(ref.stats[metricKey].mean);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (metricKey === "r2" && lo < -1) lo = -1;
    const pad = (hi - lo) * 0.12 || Math.max(Math.abs(hi) * 0.05, 0.01);
    lo -= pad; hi += pad;
    const ih = height - M.t - M.b, iw = width - M.l - M.r;
    const y = v => M.t + ih * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo));
    const band = iw / models.length, cx = i => M.l + band * (i + 0.5);
    for (const t of niceTicks(lo, hi, Math.max(3, Math.round(ih / 55)))) {
      svg.append(el("svg:line", { class: "gridline", x1: M.l, x2: width - M.r, y1: y(t), y2: y(t) }));
      svg.append(el("svg:text", { class: "tick", x: M.l - 8, y: y(t) + 4, "text-anchor": "end" }, fmt(t)));
    }
    svg.append(el("svg:line", { class: "baseline", x1: M.l, x2: width - M.r, y1: M.t + ih, y2: M.t + ih }));
    if (ref) {
      const ry = y(ref.stats[metricKey].mean);
      svg.append(el("svg:line", { class: "reflin", x1: M.l, x2: width - M.r, y1: ry, y2: ry }));
      if (!narrow) {
        svg.append(el("svg:text", { class: "reflab", x: width - M.r + 6, y: ry - 2 }, "Constant"));
        svg.append(el("svg:text", { class: "reflab", x: width - M.r + 6, y: ry + 11 }, "guess"));
      }
    }
    models.forEach((m, i) => {
      const s = m.stats[metricKey], g = el("svg:g", { class: "col" }), c = "c-" + m.family, x = cx(i);
      const hit = el("svg:rect", { class: "hit", x: x - band / 2 + 2, y: M.t, width: band - 4, height: ih + M.b - 4, rx: 6,
        tabindex: 0, "aria-label": `${m.label}: ${fmt(s.mean)}` });
      g.append(hit);
      if (spread(s)) g.append(el("svg:line", { class: "whisk " + c, x1: x, x2: x, y1: y(s.mean - s.sd), y2: y(s.mean + s.sd) }));
      const half = Math.min(22, band * 0.3);
      g.append(el("svg:line", { class: "meanbar " + c, x1: x - half, x2: x + half, y1: y(s.mean), y2: y(s.mean) }));
      const same = s.seeds.every(v => v === s.seeds[0]);      // deterministic model: one mark, not a smear
      const seeds = same ? [s.seeds[0]] : s.seeds, n = seeds.length, jit = Math.min(16, band * 0.18);
      seeds.forEach((v, j) => {
        const dx = n > 1 ? -jit + (2 * jit * j) / (n - 1) : 0;
        if (m.marker === "square") {
          g.append(el("svg:rect", { class: "dot " + c, x: x + dx - 4.6, y: y(v) - 4.6, width: 9.2, height: 9.2, rx: 1 }));
        } else if (m.marker === "diamond") {
          const sz = 5.2;
          g.append(el("svg:rect", { class: "dot " + c, x: x + dx - sz, y: y(v) - sz, width: 2 * sz, height: 2 * sz,
            transform: `rotate(45 ${x + dx} ${y(v)})`, rx: 1 }));
        } else {
          g.append(el("svg:circle", { class: "dot " + c, cx: x + dx, cy: y(v), r: 5 }));
        }
      });
      wrapLabel(m.label, narrow).forEach((line, k) => g.append(el("svg:text", { class: "xl", x: x, y: M.t + ih + 16 + k * 14,
        "text-anchor": "middle", style: narrow ? "font-size:10.5px" : null }, line)));
      const on = ev => { g.classList.add("on"); const p = ev.clientX !== undefined && ev.type !== "focus" ? ev : (() => { const b = hit.getBoundingClientRect(); return { clientX: b.right, clientY: b.top + 20 }; })(); showTip(m, metricKey, task, p.clientX, p.clientY); };
      const off = () => { g.classList.remove("on"); hideTip(); };
      hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", off);
      hit.addEventListener("focus", on); hit.addEventListener("blur", off);
      svg.append(g);
    });
    container.append(svg);
  }

  // ------------------------------------------------------------ sections
  function renderCards() {
    const box = document.getElementById("cards");
    box.replaceChildren();
    for (const t of TASKS) {
      const v = t.verdict, card = el("button", { class: "card", type: "button" });
      card.append(el("div", { class: "task" }, t.name), el("div", { class: "q" }, t.question || t.title));
      if (v) {
        const nums = el("div", { class: "nums" });
        const a = el("div"); a.append(el("div", { class: "big" }, fmt(v.best_mean)), el("div", { class: "lab" }, v.best));
        const b = el("div"); b.append(el("div", { class: "big muted" }, fmt(v.gbm_mean)), el("div", { class: "lab" }, "GBM"));
        nums.append(a, b);
        if (v.graph_gbm !== undefined && state.nbr) {
          const c = el("div"); c.append(el("div", { class: "big muted" }, fmt(v.graph_gbm_mean)), el("div", { class: "lab" }, v.graph_gbm));
          nums.append(c);
        }
        const vd = el("div", { class: "verdict" });
        vd.append(el("b", null, v.best), document.createTextNode(` is ${v.word} the GBM (${fmtGain(v.gain, v.gbm_mean)} ${t.metric_names[t.primary]}).`));
        card.append(nums, vd);
      } else {
        card.append(el("div", { class: "small" }, "No GBM row to compare against."));
      }
      card.addEventListener("click", () => { selectTask(t.stem); document.getElementById("explore-h").scrollIntoView({ behavior: "smooth" }); });
      box.append(card);
    }
  }
  function renderPanels() {
    const box = document.getElementById("panels");
    box.replaceChildren();
    for (const t of TASKS) {
      const p = el("div", { class: "panel" }), head = el("div", { class: "head" });
      const title = el("div"); title.append(el("h3", null, t.name), el("div", { class: "small" }, t.metric_names[t.primary]));
      const more = el("button", { type: "button" }, "Explore →");
      more.addEventListener("click", () => { selectTask(t.stem); document.getElementById("explore-h").scrollIntoView({ behavior: "smooth" }); });
      head.append(title, more);
      const chart = el("div", { class: "chart" });
      p.append(head, chart);
      box.append(p);
      dotPlot(chart, t, t.primary, 230);
    }
  }
  function renderExplore() {
    const t = byStem[state.task];
    if (!t) return;
    const ms = document.getElementById("sel-metric");
    if (!t.metrics.includes(state.metric)) state.metric = t.primary;
    ms.replaceChildren(...t.metrics.map(k => { const o = el("option", { value: k }, t.metric_names[k]); if (k === state.metric) o.selected = true; return o; }));
    document.getElementById("sel-task").value = t.stem;
    document.getElementById("ex-q").textContent = t.question || t.name;
    document.getElementById("ex-setup").textContent = t.setup;
    dotPlot(document.getElementById("ex-chart"), t, state.metric, 340);
    document.getElementById("ex-help").textContent = DATA.metric_help[state.metric] || "";
    // table
    const tab = document.getElementById("ex-table");
    tab.replaceChildren();
    const rows = t.models.filter(m => (state.nbr || !m.nbr) && (state.ref || !m.reference));
    const thead = el("thead"), hr = el("tr");
    hr.append(el("th", null, "Model"));
    for (const k of t.metrics) hr.append(el("th", k === state.metric ? { class: "sel" } : null, t.metric_names[k]));
    hr.append(el("th", null, "Parameters"));
    thead.append(hr);
    const tbody = el("tbody");
    const best = {};
    for (const k of t.metrics) {
      const cand = rows.filter(m => !m.reference && m.stats[k]).map(m => m.stats[k].mean);
      if (cand.length) best[k] = t.lower.includes(k) ? Math.min(...cand) : Math.max(...cand);
    }
    for (const m of rows) {
      const tr = el("tr");
      tr.append(el("td", null, m.label));
      for (const k of t.metrics) {
        const s = m.stats[k], td = el("td", s && !m.reference && Math.abs(s.mean - best[k]) < 1e-12 ? { class: "best" } : null);
        if (s) { td.append(document.createTextNode(fmt(s.mean))); if (spread(s)) td.append(el("span", { class: "sd" }, " ± " + fmtSd(s.sd, s.mean))); }
        else td.textContent = "–";
        tr.append(td);
      }
      tr.append(el("td", null, m.params ? m.params.toLocaleString() : "–"));
      tbody.append(tr);
    }
    tab.append(thead, tbody);
    const st = Object.entries(t.settings).map(([k, v]) => `${k} ${v}`).join(", ");
    document.getElementById("ex-meta").textContent =
      `${t.title}. Bold = best model in each column. ${t.n_seeds} seed${t.n_seeds > 1 ? "s" : ""}. Shared graph-model settings: ${st}.`;
  }
  function selectTask(stem) { state.task = stem; state.metric = byStem[stem].primary; renderExplore(); }
  function renderAll() { renderCards(); renderPanels(); renderExplore(); }

  // ------------------------------------------------------------ wiring
  const selTask = document.getElementById("sel-task");
  selTask.append(...TASKS.map(t => el("option", { value: t.stem }, t.name)));
  selTask.addEventListener("change", e => selectTask(e.target.value));
  document.getElementById("sel-metric").addEventListener("change", e => { state.metric = e.target.value; renderExplore(); });
  document.getElementById("f-nbr").addEventListener("change", e => { state.nbr = e.target.checked; renderAll(); });
  document.getElementById("f-ref").addEventListener("change", e => { state.ref = e.target.checked; renderAll(); });
  const themeBtn = document.getElementById("theme");
  function isDark() {
    const t = document.documentElement.getAttribute("data-theme");
    return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncThemeLabel() { themeBtn.textContent = isDark() ? "Light theme" : "Dark theme"; }
  themeBtn.addEventListener("click", () => { document.documentElement.setAttribute("data-theme", isDark() ? "light" : "dark"); syncThemeLabel(); });
  syncThemeLabel();
  document.getElementById("foot").textContent = `Generated ${DATA.generated} by make_report.py from results/*.json.`;
  renderAll();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
