"""Belgian underwriting-fraud network (Kaggle: brunordeprez/underwriting-fraud-real-worldanonymised)
-> two snapshot graphs, t0 (train) and t1 (test), for out-of-time fraud classification of companies
or administrators.

  python preprocess_underwriting.py --raw-dir data/raw/underwriting --out data/processed/underwriting.pt

Source: anonymised Allianz Benelux non-life underwriting records (CC BY-NC 4.0). Cite
Vandervorst, Deprez, Verbeke & Verdonck (2026), DMKD 40(5):78, doi:10.1007/s10618-026-01249-0.

Graph definition (one homogeneous PyG `Data` per snapshot; the node type is a feature)
  nodes        companies and administrators (directors / statutory officers)
  edges        company-administrator (a registered mandate) and company-company; undirected, unweighted
  edge_type    0 = company-administrator, 1 = company-company
  node_type    0 = company, 1 = administrator
  x            [is_company, company block, administrator block]; the other type's block is zero
                 company:  legal form (one-hot, top --top-legal + other), district (one-hot, top
                           --top-district + other), log #companies in the district, log1p workforce counts
                           (employees, servants, directors, workers), no-workforce flag
                 admin:    age bucket (one-hot, 0 = unknown), gender (one-hot)
  y            fraud label (1 / 0), -1 = unlabelled. Company and administrator labels are separate
               problems: <3% of labelled nodes of one type touch a labelled node of the other type.
  masks        per target type T in {company, admin}:
                 t0: train_mask_T / val_mask_T   stratified --val-frac split of the t0-labelled T nodes
                 t1: test_mask_T                  t1-labelled T nodes that were NOT labelled at t0
               88% of the administrators (and 29% of the companies) labelled at t1 are the same nodes,
               with the same label, as at t0; scoring them again would re-test training examples.

Data problems handled here (found while profiling the raw files)
  * categorical codes were label-encoded separately per snapshot (e.g. legal form 73 at t0 is 77 at t1).
    t1 codes are mapped to t0 codes by majority vote over the companies present in both snapshots
    (ids and features only, never labels); codes never seen at t0 fall into "other".
  * 1,664 administrator rows are exact duplicates within a snapshot; they are dropped.
  * 60 company-company self-loops are dropped.
Size: each snapshot keeps only the --hops-hop neighbourhood of its labelled nodes (~150k of ~3M nodes).
With --hops = layers + 1 every node within `layers` hops keeps all of its edges, so a GNN with up to
`layers` layers computes exactly what it would on the full snapshot.
Leakage notes: vocabularies, district counts and scalers come from t0 only; labels are never features.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, k_hop_subgraph, remove_self_loops, to_undirected

from graph_utils import Standardizer, index_to_mask, save_processed, stratified_split

CAT_COLS = {"Legal Form": "legal", "Localisation district category": "district"}
WORK_COLS = ["EmployeesNumber", "WorkersServantsNumber", "WorkersDirectorsNumber", "WorkersWorkersNumber"]
AGE_BUCKETS, GENDERS = range(7), range(3)
TARGETS = {"company": 0, "admin": 1}


# --------------------------------- loading ---------------------------------
def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    paths = {k: raw_dir / f"{k}.csv" for k in ("company_nodes", "admin_nodes", "edges")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}. Download the Kaggle dataset "
                                "(brunordeprez/underwriting-fraud-real-worldanonymised) and unzip it there.")
    comp = pd.read_csv(paths["company_nodes"])
    admin = pd.read_csv(paths["admin_nodes"])
    edges = pd.read_csv(paths["edges"])
    n = len(admin)
    admin = admin.drop_duplicates()
    if admin.duplicated(["admin_id", "split"]).any():
        raise ValueError("administrator rows with the same id and snapshot but different values")
    print(f"dropped {n - len(admin)} exact duplicate administrator rows")
    return comp, admin, edges


# ---------------------------- code alignment (t1 -> t0) ----------------------------
def align_codes(comp: pd.DataFrame, col: str):
    """Map every t1 code of `col` to the t0 code most of the same companies had. Returns (mapping, purity)."""
    t0 = comp[comp["split"] == "train"].set_index("company_id")[col]
    t1 = comp[comp["split"] == "test"].set_index("company_id")[col]
    both = t0.index.intersection(t1.index)
    ct = pd.crosstab(t1.loc[both], t0.loc[both])
    mapping = ct.idxmax(axis=1)
    purity = float(ct.max(axis=1).sum() / ct.values.sum())
    return mapping, purity


# ------------------------------ node features ------------------------------
def one_hot(values: pd.Series, vocab, prefix):
    cols = {f"{prefix}_{v}": (values == v).astype(np.float32) for v in vocab}
    cols[f"{prefix}_other"] = (~values.isin(vocab)).astype(np.float32)
    return pd.DataFrame(cols, index=values.index)


def company_features(comp: pd.DataFrame, vocab: dict, district_counts: pd.Series) -> pd.DataFrame:
    parts = [one_hot(comp["legal"], vocab["legal"], "legal"),
             one_hot(comp["district"], vocab["district"], "district")]
    f = pd.concat(parts, axis=1)
    f["log_district_companies"] = np.log1p(comp["district"].map(district_counts).fillna(0)).astype(np.float32)
    for c in WORK_COLS:
        f[f"log1p_{c}"] = np.log1p(comp[c].clip(lower=0)).astype(np.float32)
    f["no_workforce"] = (comp[WORK_COLS].sum(axis=1) == 0).astype(np.float32)
    return f


def admin_features(admin: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([one_hot(admin["age_bucket"], list(AGE_BUCKETS), "age"),
                      one_hot(admin["Gender"], list(GENDERS), "gender")], axis=1).drop(
        columns=["age_other", "gender_other"])


# -------------------------------- snapshot --------------------------------
def snapshot(comp, admin, edges, fc, fa, hops):
    """One snapshot as a Data object, restricted to the `hops`-hop neighbourhood of its labelled nodes."""
    ids = np.concatenate([comp["company_id"].to_numpy(), admin["admin_id"].to_numpy()])
    index = pd.Series(np.arange(len(ids)), index=ids)
    nc, na = len(comp), len(admin)
    x = np.zeros((nc + na, 1 + fc.shape[1] + fa.shape[1]), dtype=np.float32)
    x[:nc, 0] = 1.0
    x[:nc, 1:1 + fc.shape[1]] = fc.to_numpy()
    x[nc:, 1 + fc.shape[1]:] = fa.to_numpy()
    y = np.concatenate([comp["label"].fillna(-1).to_numpy(), admin["label"].fillna(-1).to_numpy()]).astype(np.int64)
    node_type = np.concatenate([np.zeros(nc), np.ones(na)]).astype(np.int64)

    src, dst = index[edges["node_id_1"]].to_numpy(), index[edges["node_id_2"]].to_numpy()
    etype = (edges["edge_type"] == "company-company").to_numpy().astype(np.int64)
    ei, et = remove_self_loops(torch.from_numpy(np.stack([src, dst])), torch.from_numpy(etype))
    ei, et = to_undirected(ei, et, num_nodes=len(ids), reduce="max")
    ei, et = coalesce(ei, et, num_nodes=len(ids), reduce="max")

    seeds = torch.from_numpy(np.flatnonzero(y >= 0))
    keep, ei, _, emask = k_hop_subgraph(seeds, hops, ei, relabel_nodes=True, num_nodes=len(ids))
    keep_np = keep.numpy()
    d = Data(x=torch.from_numpy(x[keep_np]), edge_index=ei, edge_type=et[emask],
             y=torch.from_numpy(y[keep_np]), node_type=torch.from_numpy(node_type[keep_np]))
    return d, ids[keep_np], len(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/underwriting")
    p.add_argument("--out", default="data/processed/underwriting.pt")
    p.add_argument("--top-legal", type=int, default=20, help="legal forms one-hot encoded (by t0 frequency)")
    p.add_argument("--top-district", type=int, default=30, help="districts one-hot encoded (by t0 frequency)")
    p.add_argument("--hops", type=int, default=3, help="neighbourhood kept around labelled nodes (layers + 1)")
    p.add_argument("--val-frac", type=float, default=0.2, help="share of t0-labelled nodes used for validation")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    comp, admin, edges = load_raw(args.raw_dir)
    comp = comp.rename(columns=CAT_COLS)
    purity = {}
    for col in CAT_COLS.values():
        mapping, purity[col] = align_codes(comp, col)
        t1 = comp["split"] == "test"
        comp.loc[t1, col] = comp.loc[t1, col].map(mapping).fillna(-1).astype(int)
        print(f"aligned t1 '{col}' codes to t0 codes ({purity[col]:.1%} of shared companies agree)")

    t0c = comp[comp["split"] == "train"]
    vocab = {"legal": t0c["legal"].value_counts().index[:args.top_legal].tolist(),
             "district": t0c["district"].value_counts().index[:args.top_district].tolist()}
    district_counts = t0c["district"].value_counts()

    snaps, node_ids, full_sizes = {}, {}, {}
    for split, name in (("train", "t0"), ("test", "t1")):
        c = comp[comp["split"] == split].reset_index(drop=True)
        a = admin[admin["split"] == split].reset_index(drop=True)
        fc, fa = company_features(c, vocab, district_counts), admin_features(a)
        snaps[name], node_ids[name], full_sizes[name] = snapshot(
            c, a, edges[edges["split"] == split], fc, fa, args.hops)
        print(f"{name}: kept {snaps[name].num_nodes:,} of {full_sizes[name]:,} nodes, "
              f"{snaps[name].edge_index.size(1) // 2:,} undirected edges")
    feature_names = ["is_company"] + list(fc.columns) + list(fa.columns)

    # Standardise the continuous company columns with t0 company statistics only.
    cont = [i for i, n in enumerate(feature_names) if n.startswith("log")]
    is_c0 = snaps["t0"].node_type == 0
    sc = Standardizer.fit(snaps["t0"].x[is_c0][:, cont].numpy())
    for d in snaps.values():
        is_c = d.node_type == 0
        d.x[is_c.nonzero(as_tuple=True)[0][:, None], torch.tensor(cont)] = torch.from_numpy(
            sc.transform(d.x[is_c][:, cont].numpy()))

    # Masks: random validation split within t0, out-of-time test on t1 nodes NOT labelled at t0.
    split_sizes = {}
    for tname, tcode in TARGETS.items():
        d0, d1 = snaps["t0"], snaps["t1"]
        lab0 = np.flatnonzero(((d0.node_type == tcode) & (d0.y >= 0)).numpy())
        tr, va, _ = stratified_split(d0.y[lab0].numpy(), (1 - args.val_frac, args.val_frac, 0.0), args.seed)
        d0[f"train_mask_{tname}"] = index_to_mask(lab0[tr], d0.num_nodes)
        d0[f"val_mask_{tname}"] = index_to_mask(lab0[va], d0.num_nodes)
        known = set(node_ids["t0"][lab0])
        lab1 = ((d1.node_type == tcode) & (d1.y >= 0)).numpy()
        repeat = np.isin(node_ids["t1"], list(known))
        d1[f"test_mask_{tname}"] = torch.from_numpy(lab1 & ~repeat)
        split_sizes[tname] = {"train": len(tr), "val": len(va), "test": int((lab1 & ~repeat).sum()),
                              "t1_repeats_excluded": int((lab1 & repeat).sum()),
                              "fraud_rate_train": float(d0.y[lab0[tr]].float().mean()),
                              "fraud_rate_test": float(d1.y[torch.from_numpy(lab1 & ~repeat)].float().mean())}
        s = split_sizes[tname]
        print(f"{tname}: train/val/test {s['train']}/{s['val']}/{s['test']} (fraud {s['fraud_rate_train']:.1%} "
              f"train, {s['fraud_rate_test']:.1%} test); {s['t1_repeats_excluded']} t1 labels repeat t0 -> excluded")
    for d in snaps.values():
        d.validate(raise_on_error=True)

    meta = {
        "dataset": "underwriting_fraud_belgium", "task": "node_classification", "targets": list(TARGETS),
        "snapshots": {"t0": "train", "t1": "test"}, "feature_names": feature_names,
        "company_feature_idx": [i for i, n in enumerate(feature_names) if not n.startswith(("age_", "gender_"))],
        "admin_feature_idx": [i for i, n in enumerate(feature_names) if n == "is_company" or n.startswith(("age_", "gender_"))],
        "vocab": {k: [int(v) for v in vs] for k, vs in vocab.items()}, "code_alignment_purity": purity,
        "standardizer": sc.to_dict(), "standardized_columns": [feature_names[i] for i in cont],
        "full_snapshot_nodes": full_sizes, "kept_nodes": {k: d.num_nodes for k, d in snaps.items()},
        "split_sizes": split_sizes, "args": vars(args),
    }
    save_processed(args.out, {"snapshots": snaps, "node_ids": {k: v.tolist() for k, v in node_ids.items()}}, meta)
    print(f"features: {len(feature_names)}   saved -> {args.out}")


if __name__ == "__main__":
    main()
