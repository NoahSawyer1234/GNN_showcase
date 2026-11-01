"""Generate tiny synthetic stand-ins for the two datasets, in the REAL file formats.

  python make_synthetic_data.py --out data/synthetic

Use them to smoke-test the preprocessing pipelines before downloading the real data:
  python preprocess_medicare.py --raw-dir data/synthetic/medicare --out data/synthetic/processed/medicare.pt
  python preprocess_wildfire.py --raw-dir data/synthetic/wildfire  --out-dir data/synthetic/processed/wildfire

The values are random with a little planted structure; they are NOT realistic data and any model
accuracy on them is meaningless.
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from preprocess_medicare import CHRONIC, DIAG, PROC


# ------------------------------------ Medicare ------------------------------------
def make_medicare(out: Path, rng, n_prov=80, n_bene=600, n_phys=120, n_claims=4000):
    out.mkdir(parents=True, exist_ok=True)
    provs = [f"PRV{51000 + i}" for i in range(n_prov)]
    fraud = rng.random(n_prov) < 0.15
    # Fraudulent providers share a small pool of physicians and benes (a "ring").
    ring_phys = rng.choice(n_phys, 12, replace=False)
    ring_bene = rng.choice(n_bene, 80, replace=False)
    pd.DataFrame({"Provider": provs, "PotentialFraud": np.where(fraud, "Yes", "No")}).to_csv(
        out / "Train-1542865627584.csv", index=False)

    bene = pd.DataFrame({
        "BeneID": [f"BENE{100000 + i}" for i in range(n_bene)],
        "DOB": pd.to_datetime("1920-01-01") + pd.to_timedelta(rng.integers(0, 365 * 35, n_bene), "D"),
        "DOD": pd.NaT, "Gender": rng.integers(1, 3, n_bene), "Race": rng.choice([1, 2, 3, 5], n_bene),
        "RenalDiseaseIndicator": rng.choice(["0", "Y"], n_bene, p=[0.85, 0.15]),
        "State": rng.integers(1, 50, n_bene), "County": rng.integers(1, 500, n_bene),
        "NoOfMonths_PartACov": 12, "NoOfMonths_PartBCov": 12})
    for c in CHRONIC:
        bene[c] = rng.choice([1, 2], n_bene, p=[0.35, 0.65])
    for c in ("IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
              "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"):
        bene[c] = rng.integers(0, 20000, n_bene)
    dead = rng.random(n_bene) < 0.05
    bene.loc[dead, "DOD"] = pd.Timestamp("2009-06-01")
    bene.to_csv(out / "Train_Beneficiarydata-1542865627584.csv", index=False)

    rows = []
    for k in range(n_claims):
        pi = int(rng.integers(n_prov))
        is_f = fraud[pi]
        b = int(rng.choice(ring_bene)) if is_f and rng.random() < 0.5 else int(rng.integers(n_bene))
        phys = [f"PHY{int(rng.choice(ring_phys) if is_f and rng.random() < 0.6 else rng.integers(n_phys))}"
                if rng.random() < pr else None for pr in (1.0, 0.3, 0.2)]
        start = pd.Timestamp("2009-01-01") + pd.Timedelta(days=int(rng.integers(0, 330)))
        inpatient = rng.random() < 0.15
        amt = float(rng.gamma(2.0, 4000 if inpatient else 300) * (1.8 if is_f else 1.0))
        row = {"BeneID": f"BENE{100000 + b}", "ClaimID": f"CLM{k}", "ClaimStartDt": start,
               "ClaimEndDt": start + pd.Timedelta(days=int(rng.integers(0, 10 if inpatient else 2))),
               "Provider": provs[pi], "InscClaimAmtReimbursed": round(amt),
               "AttendingPhysician": phys[0], "OperatingPhysician": phys[1], "OtherPhysician": phys[2],
               "ClmAdmitDiagnosisCode": "78659", "DeductibleAmtPaid": 1068 if inpatient else 0,
               "_ip": inpatient}
        for i, c in enumerate(DIAG):
            row[c] = str(int(rng.integers(1000, 9999))) if rng.random() < 0.7 ** (i + 1) + 0.1 else None
        for i, c in enumerate(PROC):
            row[c] = float(rng.integers(100, 999)) if rng.random() < 0.15 / (i + 1) else None
        if inpatient:
            row.update({"AdmissionDt": start, "DischargeDt": row["ClaimEndDt"], "DiagnosisGroupCode": "300"})
        rows.append(row)
    df = pd.DataFrame(rows)
    ip = df[df["_ip"]].drop(columns="_ip")
    op = df[~df["_ip"]].drop(columns=["_ip", "AdmissionDt", "DischargeDt", "DiagnosisGroupCode"],
                             errors="ignore")
    ip.to_csv(out / "Train_Inpatientdata-1542865627584.csv", index=False, date_format="%Y-%m-%d")
    op.to_csv(out / "Train_Outpatientdata-1542865627584.csv", index=False, date_format="%Y-%m-%d")


# ------------------------------------- Wildfire -------------------------------------
def _crc32c_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return table


_CRC = _crc32c_table()


def _masked_crc(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c = _CRC[(c ^ b) & 0xFF] ^ (c >> 8)
    c ^= 0xFFFFFFFF
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _enc_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _enc_varint(field << 3 | 2) + _enc_varint(len(payload)) + payload


def encode_example(features: dict) -> bytes:
    """features: name -> float32 array. Produces tf.train.Example wire bytes (packed FloatList)."""
    entries = b""
    for name, arr in features.items():
        float_list = _ld(1, np.asarray(arr, dtype="<f4").tobytes())
        feature = _ld(2, float_list)
        entries += _ld(1, _ld(1, name.encode()) + _ld(2, feature))
    return _ld(1, entries)


def write_tfrecord(path: Path, examples):
    with open(path, "wb") as f:
        for ex in examples:
            length = struct.pack("<Q", len(ex))
            f.write(length + struct.pack("<I", _masked_crc(length)))
            f.write(ex + struct.pack("<I", _masked_crc(ex)))


def make_wildfire(out: Path, rng, side=64, n_per_split=(24, 8, 8), files_per_split=2):
    out.mkdir(parents=True, exist_ok=True)
    smooth = lambda s: gaussian_filter(rng.normal(size=(side, side)), s)
    for split, n in zip(("train", "eval", "test"), n_per_split):
        examples = []
        for _ in range(n):
            prev = np.zeros((side, side), np.float32)
            r, c = rng.integers(10, side - 10, 2)
            prev[r - 2:r + 3, c - 2:c + 3] = 1
            wind = rng.uniform(0, 360)                          # degrees clockwise from north
            dr, dc = -np.cos(np.deg2rad(wind)), np.sin(np.deg2rad(wind))
            nxt = np.roll(prev, (int(round(2 * dr)), int(round(2 * dc))), axis=(0, 1)) + prev
            nxt = (nxt > 0).astype(np.float32)
            nxt[smooth(4) > 1.2 * smooth(4).std()] = -1         # "uncertain" pixels
            feats = {
                "elevation": 500 + 300 * smooth(6), "th": np.full((side, side), wind) + 5 * smooth(3),
                "vs": 3 + np.abs(smooth(4)), "tmmn": 280 + 3 * smooth(5), "tmmx": 295 + 3 * smooth(5),
                "sph": 0.006 + 0.001 * smooth(5), "pr": np.abs(smooth(3)),
                "pdsi": smooth(8), "NDVI": 4000 + 1500 * smooth(5), "population": np.abs(20 * smooth(2)),
                "erc": 40 + 10 * smooth(6), "PrevFireMask": prev, "FireMask": nxt}
            feats["NDVI"][rng.random((side, side)) < 0.01] = np.nan   # sprinkle missing values
            examples.append(encode_example({k: v.ravel() for k, v in feats.items()}))
        for i, chunk in enumerate(np.array_split(np.arange(n), files_per_split)):
            write_tfrecord(out / f"next_day_wildfire_spread_{split}_{i:02d}.tfrecord",
                           [examples[j] for j in chunk])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out, rng = Path(args.out), np.random.default_rng(args.seed)
    make_medicare(out / "medicare", rng)
    make_wildfire(out / "wildfire", rng)
    print(f"synthetic data written to {out}")


if __name__ == "__main__":
    main()
