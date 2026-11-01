"""GDELT Events -> a chronological sequence of country-interaction graphs ("snapshots").

  python download_gdelt.py --start 2025-01-01 --end 2025-03-31 --out data/raw/gdelt
  python preprocess_gdelt.py --raw-dir data/raw/gdelt --out data/processed/gdelt.pt --freq 1D

Forecasting setup (leakage-free by construction)
  For every target window t the snapshot holds
     INPUT   graph + node features built ONLY from the `--lookback` windows before t
     TARGETS computed from window t itself
  Nodes are countries (fixed vocabulary = top-K most active countries in the TRAINING period,
  same indices in every snapshot). One event = (Actor1CountryCode, Actor2CountryCode).

Per-snapshot `Data` fields
  x                [N, 13]  node features from the lookback window (see NODE_FEATURES)
  edge_index       [2, E]   undirected country pairs that interacted in the lookback window
  edge_weight      [E]      log1p(#events)         (positive; usable by GCNConv)
  edge_attr        [E, 2]   [log1p(#events), mean log1p(NumMentions)]

  -- node-level targets (window t) --
  y_node_cls       [N]      dominant QuadClass (0..3, see QUAD_NAMES), -1 if too few events
  y_node_reg       [N, 3]   [mean Goldstein, mean AvgTone, conflict share]
  node_target_mask [N]      True where the node had >= --min-node-events events in window t

  -- edge-level targets (window t) --
  edge_label_index [2, L]   country pairs with >= --min-label-events events in window t
                            (positives for link prediction; sample negatives while training)
  edge_label_class [L]      dominant QuadClass of the pair's events   (edge classification)
  edge_label_reg   [L, 2]   [mean Goldstein, mean AvgTone]
  edge_label_count [L]      #events
  edge_label_is_new[L]      True if the pair did NOT interact in the lookback window
                            (use to evaluate genuinely new links rather than persistence)

  -- graph-level targets (window t, all events worldwide) --
  y_graph          [1, 3]   [mean Goldstein, mean AvgTone, conflict share]
  y_graph_cls      [1]      1 if conflict share > TRAIN-period median else 0

QuadClass: 1 verbal cooperation, 2 material cooperation, 3 verbal conflict, 4 material conflict.
Domestic events (Actor1 country == Actor2 country) are not edges; they feed node features.
Chronological split: first --train-frac of windows = train, next --val-frac = val, rest = test.
Standardisation (nodes + edge_attr) and the vocabulary are fit on training windows only.

Supported raw formats: GDELT 1.0 events (58 tab-separated columns, DATEADDED = YYYYMMDD) and
GDELT 2.0 events (61 columns, DATEADDED = YYYYMMDDHHMMSS). Files are headerless, zipped TSV.
Events are assigned to windows by DATEADDED (when GDELT recorded them), falling back to SQLDATE.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graph_utils import Standardizer, save_processed

QUAD_NAMES = ["verbal_cooperation", "material_cooperation", "verbal_conflict", "material_conflict"]
NODE_FEATURES = ["log_events_as_actor1", "log_events_as_actor2", "log_domestic_events",
                 "log_partners_out", "log_partners_in", "mean_tone", "mean_goldstein",
                 "share_verbal_coop", "share_material_coop", "share_verbal_conflict",
                 "share_material_conflict", "mean_log_mentions", "active"]
SUMS = ["n", "tone", "gold", "lm", "q0", "q1", "q2", "q3"]     # additive per-row statistics
NODE_REG_TARGETS = ["goldstein", "tone", "conflict_share"]


# ------------------------------ raw file parsing ------------------------------
def gdelt_columns(n_cols: int):
    actor = ["Code", "Name", "CountryCode", "KnownGroupCode", "EthnicCode", "Religion1Code",
             "Religion2Code", "Type1Code", "Type2Code", "Type3Code"]
    geo_v1 = ["Type", "FullName", "CountryCode", "ADM1Code", "Lat", "Long", "FeatureID"]
    geo_v2 = ["Type", "FullName", "CountryCode", "ADM1Code", "ADM2Code", "Lat", "Long", "FeatureID"]
    if n_cols == 58:
        geo = geo_v1
    elif n_cols == 61:
        geo = geo_v2
    else:
        raise ValueError(f"Unexpected GDELT column count {n_cols} (expected 58 for v1 or 61 for v2)")
    cols = ["GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate"]
    cols += [f"Actor1{c}" for c in actor] + [f"Actor2{c}" for c in actor]
    cols += ["IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
             "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]
    for prefix in ("Actor1Geo_", "Actor2Geo_", "ActionGeo_"):
        cols += [prefix + g for g in geo]
    cols += ["DATEADDED", "SOURCEURL"]
    assert len(cols) == n_cols
    return cols


WANTED = ["SQLDATE", "Actor1CountryCode", "Actor2CountryCode", "QuadClass", "GoldsteinScale",
          "NumMentions", "AvgTone", "DATEADDED"]


def read_events(path) -> pd.DataFrame:
    head = pd.read_csv(path, sep="\t", header=None, nrows=1, dtype=str, quoting=3)
    cols = gdelt_columns(head.shape[1])
    return pd.read_csv(path, sep="\t", header=None, names=cols, usecols=WANTED, dtype=str,
                       quoting=3, on_bad_lines="skip")


def event_time(df: pd.DataFrame) -> pd.Series:
    added = df["DATEADDED"]
    t14 = pd.to_datetime(added.where(added.str.len() == 14), format="%Y%m%d%H%M%S", errors="coerce")
    t8 = pd.to_datetime(added.where(added.str.len() == 8), format="%Y%m%d", errors="coerce")
    sql = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
    return t14.fillna(t8).fillna(sql)


class CountryIndexer:
    """Assigns a stable integer id to every country code seen; missing/blank -> -1."""

    def __init__(self):
        self.ids = {}

    def encode(self, s: pd.Series) -> np.ndarray:
        s = s.fillna("").str.strip()
        for code in s.unique():
            if code and code not in self.ids:
                self.ids[code] = len(self.ids)
        return s.map(self.ids).fillna(-1).astype(np.int32).to_numpy()

    @property
    def codes(self):
        return [c for c, _ in sorted(self.ids.items(), key=lambda kv: kv[1])]


def aggregate_file(df: pd.DataFrame, freq: str, indexer: CountryIndexer):
    t = event_time(df)
    quad = pd.to_numeric(df["QuadClass"], errors="coerce")
    gold = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    tone = pd.to_numeric(df["AvgTone"], errors="coerce")
    ment = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1).clip(lower=0)
    ok = (t.notna() & quad.between(1, 4) & gold.notna() & tone.notna()).to_numpy()
    if not ok.any():
        return None, None
    w = pd.DataFrame({
        "t": t[ok].dt.floor(freq).to_numpy(),
        "a1": indexer.encode(df.loc[ok, "Actor1CountryCode"]),
        "a2": indexer.encode(df.loc[ok, "Actor2CountryCode"]),
        "n": 1.0, "tone": tone[ok].to_numpy(), "gold": gold[ok].to_numpy(),
        "lm": np.log1p(ment[ok]).to_numpy()})
    q = quad[ok].astype(int).to_numpy() - 1
    for k in range(4):
        w[f"q{k}"] = (q == k).astype(np.float64)
    glob = w.groupby("t")[SUMS].sum()                         # every event, for graph-level targets
    pairs = w[(w["a1"] >= 0) | (w["a2"] >= 0)].groupby(["t", "a1", "a2"], as_index=False)[SUMS].sum()
    return pairs, glob


def _regroup(df):
    return df.groupby(["a1", "a2"], as_index=False)[SUMS].sum()


def ingest(files, freq):
    indexer = CountryIndexer()
    pair_tabs, glob_tabs = {}, {}
    for i, f in enumerate(files, 1):
        pairs, glob = aggregate_file(read_events(f), freq, indexer)
        if pairs is None:
            continue
        for t, g in pairs.groupby("t"):
            g = g.drop(columns="t")
            pair_tabs[t] = g if t not in pair_tabs else _regroup(pd.concat([pair_tabs[t], g]))
        for t, row in glob.iterrows():
            glob_tabs[t] = row if t not in glob_tabs else glob_tabs[t] + row
        if i % 50 == 0 or i == len(files):
            print(f"  read {i}/{len(files)} files, {len(pair_tabs)} windows")
    return pair_tabs, glob_tabs, indexer


# ------------------------------- graph building -------------------------------
def _lookup(lut, ids):
    out = lut[np.clip(ids, 0, None)]
    out[ids < 0] = -1
    return out


def remap(tab: pd.DataFrame, lut: np.ndarray) -> pd.DataFrame:
    """Country ids -> vocabulary indices (-1 if outside the vocabulary); drop rows with no known actor."""
    t = tab.assign(a1=_lookup(lut, tab["a1"].to_numpy()), a2=_lookup(lut, tab["a2"].to_numpy()))
    return _regroup(t[(t["a1"] >= 0) | (t["a2"] >= 0)])


def involvement(tab: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Per-node sums of SUMS over every event involving the node (domestic events counted once)."""
    a1, a2 = tab["a1"].to_numpy(), tab["a2"].to_numpy()
    vals = tab[SUMS].to_numpy(np.float64)
    out = np.zeros((n_nodes, len(SUMS)))
    m1, m2 = a1 >= 0, (a2 >= 0) & (a2 != a1)
    np.add.at(out, a1[m1], vals[m1])
    np.add.at(out, a2[m2], vals[m2])
    return out


def node_features(tab: pd.DataFrame, n_nodes: int) -> np.ndarray:
    a1, a2, n = tab["a1"].to_numpy(), tab["a2"].to_numpy(), tab["n"].to_numpy()
    inv = involvement(tab, n_nodes)
    safe = np.maximum(inv[:, 0], 1.0)
    dom, out_, in_ = (np.zeros(n_nodes) for _ in range(3))
    d = (a1 >= 0) & (a1 == a2)
    np.add.at(dom, a1[d], n[d])
    m = (a1 >= 0) & (a2 != a1)
    np.add.at(out_, a1[m], n[m])
    m = (a2 >= 0) & (a1 != a2)
    np.add.at(in_, a2[m], n[m])
    pm = (a1 >= 0) & (a2 >= 0) & (a1 != a2)               # rows are unique (a1, a2) pairs
    p_out = np.bincount(a1[pm], minlength=n_nodes)
    p_in = np.bincount(a2[pm], minlength=n_nodes)
    feats = np.column_stack([
        np.log1p(out_), np.log1p(in_), np.log1p(dom), np.log1p(p_out), np.log1p(p_in),
        inv[:, 1] / safe, inv[:, 2] / safe, inv[:, 4:8] / safe[:, None],
        inv[:, 3] / safe, (inv[:, 0] > 0).astype(float)])
    assert feats.shape[1] == len(NODE_FEATURES)
    return feats.astype(np.float32)


def node_targets(tab: pd.DataFrame, n_nodes: int, min_events: int):
    inv = involvement(tab, n_nodes)
    mask = inv[:, 0] >= min_events
    safe = np.maximum(inv[:, 0], 1.0)
    reg = np.column_stack([inv[:, 2] / safe, inv[:, 1] / safe, (inv[:, 6] + inv[:, 7]) / safe])
    cls = inv[:, 4:8].argmax(1)
    reg[~mask] = 0.0
    cls[~mask] = -1
    return reg.astype(np.float32), cls.astype(np.int64), mask


def undirected_pairs(tab: pd.DataFrame) -> pd.DataFrame:
    a1, a2 = tab["a1"].to_numpy(), tab["a2"].to_numpy()
    m = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
    df = pd.DataFrame({"lo": np.minimum(a1[m], a2[m]), "hi": np.maximum(a1[m], a2[m]),
                       **{c: tab[c].to_numpy()[m] for c in SUMS}})
    return df.groupby(["lo", "hi"], as_index=False)[SUMS].sum()


def snapshot(in_tab, tgt_tab, tgt_glob, n_nodes, args) -> Data:
    # ---- inputs (lookback window) ----
    in_pairs = undirected_pairs(in_tab)
    p = in_pairs[in_pairs["n"] >= args.min_pair_events]
    lo, hi = p["lo"].to_numpy(), p["hi"].to_numpy()
    logn = np.log1p(p["n"].to_numpy())
    mlm = p["lm"].to_numpy() / np.maximum(p["n"].to_numpy(), 1)
    edge_index = np.stack([np.concatenate([lo, hi]), np.concatenate([hi, lo])]).astype(np.int64)
    attr = np.column_stack([logn, mlm])
    # ---- node targets ----
    reg, cls, mask = node_targets(tgt_tab, n_nodes, args.min_node_events)
    # ---- edge targets ----
    tp = undirected_pairs(tgt_tab)
    tp = tp[tp["n"] >= args.min_label_events]
    tlo, thi = tp["lo"].to_numpy(), tp["hi"].to_numpy()
    safe = np.maximum(tp["n"].to_numpy(), 1)
    in_keys = lo.astype(np.int64) * n_nodes + hi
    is_new = ~np.isin(tlo.astype(np.int64) * n_nodes + thi, in_keys)
    # ---- graph targets ----
    gn = max(float(tgt_glob["n"]), 1.0)
    y_graph = np.array([[tgt_glob["gold"] / gn, tgt_glob["tone"] / gn,
                         (tgt_glob["q2"] + tgt_glob["q3"]) / gn]], dtype=np.float32)

    T = torch.from_numpy
    return Data(
        x=T(node_features(in_tab, n_nodes)),
        edge_index=T(edge_index),
        edge_weight=T(np.concatenate([logn, logn]).astype(np.float32)),
        edge_attr=T(np.concatenate([attr, attr]).astype(np.float32)),
        y_node_cls=T(cls), y_node_reg=T(reg), node_target_mask=T(mask),
        edge_label_index=T(np.stack([tlo, thi]).astype(np.int64)),
        edge_label_class=T(tp[["q0", "q1", "q2", "q3"]].to_numpy().argmax(1).astype(np.int64)
                           if len(tp) else np.zeros(0, dtype=np.int64)),
        edge_label_reg=T(np.column_stack([tp["gold"] / safe, tp["tone"] / safe]).astype(np.float32)
                         if len(tp) else np.zeros((0, 2), dtype=np.float32)),
        edge_label_count=T(tp["n"].to_numpy().astype(np.float32)),
        edge_label_is_new=T(is_new),
        y_graph=T(y_graph), num_nodes=n_nodes)


# ----------------------------------- main -----------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/gdelt")
    p.add_argument("--pattern", default="*.export.CSV*", help="glob for raw event files")
    p.add_argument("--out", default="data/processed/gdelt.pt")
    p.add_argument("--freq", default="1D", help="window size: fixed pandas freq, e.g. 1h, 6h, 1D, 7D")
    p.add_argument("--lookback", type=int, default=1, help="#windows of history used as input")
    p.add_argument("--top-k", type=int, default=150, help="#countries (nodes)")
    p.add_argument("--min-pair-events", type=int, default=1, help="input edge threshold")
    p.add_argument("--min-label-events", type=int, default=3, help="edge-label threshold")
    p.add_argument("--min-node-events", type=int, default=5, help="node-target threshold")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-files", type=int, default=None, help="debug: read only the first N files")
    args = p.parse_args()

    files = sorted(Path(args.raw_dir).glob(args.pattern))[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.raw_dir}. "
                                "Run download_gdelt.py first.")
    print(f"ingesting {len(files)} files ...")
    pair_tabs, glob_tabs, indexer = ingest(files, args.freq)

    windows = pd.date_range(min(pair_tabs), max(pair_tabs), freq=args.freq)
    nw = len(windows)
    b_train, b_val = int(args.train_frac * nw), int((args.train_frac + args.val_frac) * nw)
    if b_train <= args.lookback or nw - b_val < 1:
        raise ValueError(f"Only {nw} windows: too few for lookback={args.lookback} and a 3-way split. "
                         "Download a longer period or use a smaller --freq.")

    # Vocabulary from the training period only.
    counts = np.zeros(len(indexer.ids))
    for t in windows[:b_train]:
        if t in pair_tabs:
            tab = pair_tabs[t]
            a1, a2, n = tab["a1"].to_numpy(), tab["a2"].to_numpy(), tab["n"].to_numpy()
            np.add.at(counts, a1[a1 >= 0], n[a1 >= 0])
            m = (a2 >= 0) & (a2 != a1)
            np.add.at(counts, a2[m], n[m])
    top = np.argsort(-counts)[: args.top_k]
    top = top[counts[top] > 0]
    lut = np.full(len(indexer.ids), -1, dtype=np.int64)
    lut[top] = np.arange(len(top))
    codes = indexer.codes
    vocab = [codes[i] for i in top]
    n_nodes = len(vocab)
    print(f"{nw} windows ({args.freq}); {n_nodes} countries; train windows: {b_train}")

    empty = pd.DataFrame({"a1": np.zeros(0, np.int64), "a2": np.zeros(0, np.int64),
                          **{c: np.zeros(0) for c in SUMS}})
    tabs = {t: (remap(pair_tabs[t], lut) if t in pair_tabs else empty) for t in windows}

    split_of = lambda i: "train" if i < b_train else ("val" if i < b_val else "test")
    snaps = {"train": [], "val": [], "test": []}
    stamps = {"train": [], "val": [], "test": []}
    for i in range(args.lookback, nw):
        t = windows[i]
        look = [tabs[windows[j]] for j in range(i - args.lookback, i)]
        in_tab = _regroup(pd.concat(look)) if len(look) > 1 else look[0]
        if t not in glob_tabs or len(tabs[t]) == 0 or len(in_tab) == 0:
            continue                                       # gap in the downloaded data
        s = split_of(i)
        snaps[s].append(snapshot(in_tab, tabs[t], glob_tabs[t], n_nodes, args))
        stamps[s].append(str(t))
    if not snaps["train"] or not snaps["val"] or not snaps["test"]:
        raise ValueError({k: len(v) for k, v in snaps.items()})

    # Fit scalers / thresholds on TRAIN snapshots only, then apply to all splits.
    x_sc = Standardizer.fit(np.concatenate([d.x.numpy() for d in snaps["train"]]))
    ea_sc = Standardizer.fit(np.concatenate([d.edge_attr.numpy() for d in snaps["train"]]))
    thr = float(np.median([d.y_graph[0, 2].item() for d in snaps["train"]]))
    for part in snaps.values():
        for d in part:
            d.x = torch.from_numpy(x_sc.transform(d.x.numpy()))
            d.edge_attr = torch.from_numpy(ea_sc.transform(d.edge_attr.numpy()))
            d.y_graph_cls = torch.tensor([int(d.y_graph[0, 2].item() > thr)])

    meta = {
        "dataset": "gdelt_country_interactions", "tasks": ["node", "edge", "graph"],
        "freq": args.freq, "lookback": args.lookback, "countries": vocab,
        "node_features": NODE_FEATURES, "quad_names": QUAD_NAMES,
        "node_reg_targets": NODE_REG_TARGETS, "edge_attr": ["log1p_events", "mean_log_mentions"],
        "graph_targets": NODE_REG_TARGETS, "graph_cls_threshold_conflict_share": thr,
        "window_start": stamps, "standardizer_x": x_sc.to_dict(), "standardizer_edge_attr": ea_sc.to_dict(),
        "args": vars(args), "num_snapshots": {k: len(v) for k, v in snaps.items()},
    }
    save_processed(args.out, snaps, meta)

    d = snaps["train"][0]
    print(f"snapshots train/val/test: {len(snaps['train'])}/{len(snaps['val'])}/{len(snaps['test'])}")
    print(f"example: nodes={d.num_nodes} x={tuple(d.x.shape)} edges={d.edge_index.size(1)} "
          f"label_edges={d.edge_label_index.size(1)} (new: {d.edge_label_is_new.float().mean():.0%})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
