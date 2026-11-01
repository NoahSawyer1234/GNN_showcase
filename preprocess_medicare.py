"""CMS Medicare provider fraud (Kaggle: rohitrox/healthcare-provider-fraud-detection-analysis)
-> ONE provider graph for node classification (PotentialFraud).

  python preprocess_medicare.py --raw-dir data/raw/medicare --out data/processed/medicare.pt

Graph definition
  nodes        providers that have labels (the *Train* files; the *Test* files carry no labels)
  node x       provider-level aggregates of their claims + the mix of beneficiaries they treat
  node y       PotentialFraud (1 = Yes, 0 = No)
  edges        two providers are linked when they share beneficiaries and/or physicians
               (attending / operating / other). Undirected (both directions stored).
  edge_weight  log1p(#shared entities)            (positive, usable by GCNConv)
  edge_attr    [log1p(#shared beneficiaries), log1p(#shared physicians)]
  masks        stratified train/val/test split of providers (the split is random, seeded)

Leakage notes: features use claims only (never labels); standardisation statistics are fit on
training nodes only; the graph itself is transductive (all providers are present at train time).

Expected files in --raw-dir (the numeric suffix in the Kaggle names is ignored):
  Train-*.csv  Train_Beneficiarydata-*.csv  Train_Inpatientdata-*.csv  Train_Outpatientdata-*.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from graph_utils import Standardizer, index_to_mask, save_processed, stratified_split

DIAG = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
PHYS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
CHRONIC = ["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
           "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
           "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
           "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke"]
# Heavy-tailed columns that get log1p before standardisation.
LOG_COLS = ["n_claims", "n_benes", "reimb_sum", "reimb_mean", "reimb_std", "reimb_max",
            "deductible_mean", "n_attending", "n_operating", "n_other", "claims_per_bene",
            "claims_per_attending", "n_unique_diag", "n_unique_proc",
            "bene_IPAnnualReimbursementAmt", "bene_OPAnnualReimbursementAmt"]


# --------------------------------- loading ---------------------------------
def _find(raw_dir: Path, prefix: str) -> Path:
    hits = sorted(raw_dir.glob(f"{prefix}*.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No file matching '{prefix}*.csv' in {raw_dir}. Download the Kaggle dataset "
            "(healthcare-provider-fraud-detection-analysis) and unzip it there.")
    return hits[0]


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    labels = pd.read_csv(_find(raw_dir, "Train-"))
    bene = pd.read_csv(_find(raw_dir, "Train_Beneficiarydata"))
    ip = pd.read_csv(_find(raw_dir, "Train_Inpatientdata"),
                     parse_dates=["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"])
    op = pd.read_csv(_find(raw_dir, "Train_Outpatientdata"),
                     parse_dates=["ClaimStartDt", "ClaimEndDt"])
    ip["is_inpatient"], op["is_inpatient"] = 1, 0
    claims = pd.concat([ip, op], ignore_index=True, sort=False)
    need = {"Provider", "PotentialFraud"} - set(labels.columns)
    if need:
        raise ValueError(f"Label file is missing columns {need}; found {list(labels.columns)}")
    return labels, bene, claims


# ------------------------------ node features ------------------------------
def beneficiary_table(bene: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    b = pd.DataFrame({"BeneID": bene["BeneID"]})
    b["age"] = (ref_date - pd.to_datetime(bene["DOB"], errors="coerce")).dt.days / 365.25
    b["is_dead"] = bene["DOD"].notna().astype(float)
    b["female"] = (bene["Gender"] == 2).astype(float)
    b["renal"] = (bene["RenalDiseaseIndicator"].astype(str) == "Y").astype(float)
    for c in CHRONIC:                       # coded 1 = has condition, 2 = does not
        b[c] = (bene[c] == 1).astype(float)
    b["n_chronic"] = b[CHRONIC].sum(axis=1)
    for c in ("IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"):
        b[c] = bene[c].astype(float)
    return b


def provider_features(claims: pd.DataFrame, bene: pd.DataFrame) -> pd.DataFrame:
    c = claims.copy()
    c["duration"] = (c["ClaimEndDt"] - c["ClaimStartDt"]).dt.days.clip(lower=0)
    c["los"] = (c["DischargeDt"] - c["AdmissionDt"]).dt.days.clip(lower=0).fillna(0)
    c["n_diag"] = c[DIAG].notna().sum(axis=1)
    c["n_proc"] = c[PROC].notna().sum(axis=1)
    c["has_operating"] = c["OperatingPhysician"].notna().astype(float)
    c["has_other"] = c["OtherPhysician"].notna().astype(float)

    g = c.groupby("Provider")
    f = pd.DataFrame({
        "n_claims": g.size(),
        "frac_inpatient": g["is_inpatient"].mean(),
        "n_benes": g["BeneID"].nunique(),
        "reimb_sum": g["InscClaimAmtReimbursed"].sum(),
        "reimb_mean": g["InscClaimAmtReimbursed"].mean(),
        "reimb_std": g["InscClaimAmtReimbursed"].std().fillna(0),
        "reimb_max": g["InscClaimAmtReimbursed"].max(),
        "deductible_mean": g["DeductibleAmtPaid"].mean(),
        "duration_mean": g["duration"].mean(),
        "los_mean": g["los"].mean(),
        "n_diag_mean": g["n_diag"].mean(),
        "n_proc_mean": g["n_proc"].mean(),
        "n_attending": g["AttendingPhysician"].nunique(),
        "n_operating": g["OperatingPhysician"].nunique(),
        "n_other": g["OtherPhysician"].nunique(),
        "frac_has_operating": g["has_operating"].mean(),
        "frac_has_other": g["has_other"].mean(),
    })
    f["claims_per_bene"] = f["n_claims"] / f["n_benes"].clip(lower=1)
    f["claims_per_attending"] = f["n_claims"] / f["n_attending"].clip(lower=1)
    for name, cols in (("n_unique_diag", DIAG), ("n_unique_proc", PROC)):
        long = c[["Provider"] + cols].melt(id_vars="Provider").dropna(subset=["value"])
        f[name] = long.groupby("Provider")["value"].nunique()
    f[["n_unique_diag", "n_unique_proc"]] = f[["n_unique_diag", "n_unique_proc"]].fillna(0)

    # Who are the patients? (mean over the provider's distinct beneficiaries)
    b = beneficiary_table(bene, c["ClaimEndDt"].max())
    pairs = c[["Provider", "BeneID"]].drop_duplicates().merge(b, on="BeneID", how="left")
    mix_cols = ["age", "is_dead", "female", "renal", "n_chronic", *CHRONIC,
                "IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"]
    f = f.join(pairs.groupby("Provider")[mix_cols].mean().add_prefix("bene_"))

    f = f.fillna(f.median(numeric_only=True)).fillna(0.0)
    for col in LOG_COLS:
        f[col] = np.log1p(f[col].clip(lower=0))
    return f


# ---------------------------------- edges ----------------------------------
def _incidence(pairs: pd.DataFrame, prov_index: pd.Series, entity_col: str, max_degree):
    rows = pairs["Provider"].map(prov_index)
    ok = rows.notna().to_numpy()
    rows = rows[ok].astype(int).to_numpy()
    cols = pairs[entity_col].astype("category").cat.codes.to_numpy()[ok]
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(prov_index), int(cols.max()) + 1 if len(cols) else 1))
    if max_degree:                          # drop hub entities (very large cliques are uninformative)
        deg = np.asarray(m.sum(axis=0)).ravel()
        m = (m @ sp.diags((deg <= max_degree).astype(np.float32))).tocsr()
    return m


def _projection(m: sp.csr_matrix) -> sp.csr_matrix:
    a = (m @ m.T).tocsr()
    a = a - sp.diags(a.diagonal())
    a.eliminate_zeros()
    return a.tocsr()


def build_edges(claims, prov_index, edge_source, min_shared, max_entity_degree):
    bene_pairs = claims[["Provider", "BeneID"]].drop_duplicates()
    phys_pairs = (claims[["Provider"] + PHYS].melt(id_vars="Provider")
                  .dropna(subset=["value"])[["Provider", "value"]].drop_duplicates())
    a_b = _projection(_incidence(bene_pairs, prov_index, "BeneID", max_entity_degree))
    a_p = _projection(_incidence(phys_pairs, prov_index, "value", max_entity_degree))
    combined = {"beneficiary": a_b, "physician": a_p, "both": a_b + a_p}[edge_source].tocoo()
    keep = combined.data >= min_shared
    row, col, w = combined.row[keep], combined.col[keep], combined.data[keep]
    nb = np.asarray(a_b[row, col]).ravel()
    npys = np.asarray(a_p[row, col]).ravel()
    edge_index = torch.from_numpy(np.stack([row, col]).astype(np.int64))
    edge_weight = torch.from_numpy(np.log1p(w).astype(np.float32))
    edge_attr = torch.from_numpy(np.column_stack([np.log1p(nb), np.log1p(npys)]).astype(np.float32))
    return edge_index, edge_weight, edge_attr


# ----------------------------------- main -----------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/medicare")
    p.add_argument("--out", default="data/processed/medicare.pt")
    p.add_argument("--edge-source", choices=["beneficiary", "physician", "both"], default="both")
    p.add_argument("--min-shared", type=float, default=1, help="min #shared entities to keep an edge")
    p.add_argument("--max-entity-degree", type=int, default=None,
                   help="ignore beneficiaries/physicians linked to more providers than this")
    p.add_argument("--split", type=float, nargs=3, default=(0.6, 0.2, 0.2), metavar=("TR", "VA", "TE"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    labels, bene, claims = load_raw(args.raw_dir)
    feats = provider_features(claims, bene)

    labels = labels.drop_duplicates("Provider").set_index("Provider")
    providers = feats.index.intersection(labels.index).sort_values()
    dropped = len(labels) - len(providers)
    feats = feats.loc[providers]
    y = (labels.loc[providers, "PotentialFraud"].astype(str).str.lower() == "yes").astype(int).to_numpy()
    claims = claims[claims["Provider"].isin(providers)]
    prov_index = pd.Series(np.arange(len(providers)), index=providers)

    edge_index, edge_weight, edge_attr = build_edges(
        claims, prov_index, args.edge_source, args.min_shared, args.max_entity_degree)

    tr, va, te = stratified_split(y, args.split, args.seed)
    scaler = Standardizer.fit(feats.to_numpy()[tr])
    x = torch.from_numpy(scaler.transform(feats.to_numpy()))
    counts = np.bincount(y, minlength=2)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr,
                y=torch.from_numpy(y).long(),
                train_mask=index_to_mask(tr, len(y)), val_mask=index_to_mask(va, len(y)),
                test_mask=index_to_mask(te, len(y)))
    data.validate(raise_on_error=True)

    deg = torch.bincount(edge_index[0], minlength=len(y))
    meta = {
        "dataset": "cms_medicare_provider_fraud", "task": "node_classification",
        "provider_ids": providers.tolist(), "feature_names": list(feats.columns),
        "num_classes": 2, "class_counts": counts.tolist(),
        "class_weight": (len(y) / (2.0 * counts)).tolist(),    # for weighted cross-entropy
        "standardizer": scaler.to_dict(), "args": vars(args),
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "providers_without_claims_dropped": int(dropped),
    }
    save_processed(args.out, data, meta)
    print(f"providers (nodes): {len(y)}  features: {x.size(1)}  fraud rate: {y.mean():.3%}")
    print(f"edges (directed count): {edge_index.size(1)}  mean degree: {deg.float().mean():.1f}  "
          f"isolated nodes: {(deg == 0).sum().item()}")
    print(f"split (train/val/test): {len(tr)}/{len(va)}/{len(te)}   saved -> {args.out}")


if __name__ == "__main__":
    main()
