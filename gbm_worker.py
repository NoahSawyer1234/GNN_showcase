"""Gradient boosting without PyTorch.

Imported in-process by compare_utils on Linux/Windows, and run as a separate process
(`python gbm_worker.py fit|predict <dir>`) on macOS, where PyTorch and the OpenMP runtime used by
scikit-learn / LightGBM segfault when they share one process. This module must never import torch.
"""
import pickle
import sys
from pathlib import Path

import numpy as np


def fit(job: dict) -> dict:
    cls, C = job["cls"], job["num_classes"]
    ytr = np.asarray(job["ytr"])
    cw = job["class_weight"]
    sw = cw[ytr.astype(np.int64)] if (cls and cw is not None) else None
    if job["backend"] == "lightgbm":
        import lightgbm as lgb
        params = dict(n_estimators=job["trees"], learning_rate=job["lr"], num_leaves=job["leaves"],
                      min_child_samples=job["min_child"], subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=job["seed"], n_jobs=-1, verbose=-1)
        model = lgb.LGBMClassifier(**params) if cls else lgb.LGBMRegressor(**params)
        model.fit(job["Xtr"], ytr, sample_weight=sw, eval_set=[(job["Xva"], job["yva"])],
                  callbacks=[lgb.early_stopping(job["patience"], verbose=False)])
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        params = dict(max_iter=job["trees"], learning_rate=job["lr"], max_leaf_nodes=job["leaves"],
                      random_state=job["seed"], early_stopping=False)
        model = HistGradientBoostingClassifier(**params) if cls else HistGradientBoostingRegressor(**params)
        model.fit(job["Xtr"], ytr, sample_weight=sw)
    return {"model": model, "cls": cls, "num_classes": C}


def predict(obj: dict, X) -> np.ndarray:
    model, cls, C = obj["model"], obj["cls"], obj["num_classes"]
    if not cls:
        return model.predict(X)
    proba = model.predict_proba(X)
    if proba.shape[1] < C:                      # a class absent from the training rows
        full = np.zeros((len(proba), C))
        full[:, model.classes_.astype(int)] = proba
        return full
    return proba


def main():
    mode, d = sys.argv[1], Path(sys.argv[2])
    if mode == "fit":
        with open(d / "job.pkl", "rb") as f:
            obj = fit(pickle.load(f))
        with open(d / "model.pkl", "wb") as f:
            pickle.dump(obj, f, protocol=4)
    elif mode == "predict":
        with open(d / "model.pkl", "rb") as f:
            obj = pickle.load(f)
        np.save(d / "out.npy", predict(obj, np.load(d / "X.npy")))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
