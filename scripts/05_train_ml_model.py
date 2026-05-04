#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import dump
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def ece_score(y_true, p, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = y_true[m].mean()
        conf = p[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)


def build_feature_set(df, include_network=True):
    num_feats = [
        c
        for c in ["speed_mean", "speed_std", "accel_mean", "accel_std", "jerk_abs_mean", "stops_count"]
        if c in df.columns
    ]
    net_feats = []
    if include_network:
        net_feats = [c for c in df.columns if c.startswith("network_") or c.startswith("prb_usage_")]
    cat_feats = [c for c in ["cell_id", "period", "policy"] if c in df.columns]
    return num_feats, net_feats, cat_feats


def fit_model(X_tr, y_tr, X_te, y_te, cat_feats, args):
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        verbose=200,
        od_type="Iter",
        od_wait=args.early_stop,
    )
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_feats if cat_feats else None,
        eval_set=(X_te, y_te),
        use_best_model=True,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ml", default="data/ml_table.parquet")
    ap.add_argument("--out_dir", default="data/models")
    ap.add_argument("--report_dir", default="reports/ml")
    ap.add_argument("--raw_out", default="data/artifacts/ml/final/rho_hat_windows_raw.parquet")
    ap.add_argument("--cal_out", default="data/artifacts/ml/final/rho_hat_windows_calibrated.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--early_stop", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.raw_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.cal_out), exist_ok=True)

    df = pd.read_parquet(args.in_ml)
    if "label" not in df.columns or "run_id" not in df.columns:
        raise SystemExit("[ERROR] El dataset ML debe contener columnas 'label' y 'run_id'.")

    num_feats, net_feats, cat_feats = build_feature_set(df, include_network=True)
    feats = num_feats + net_feats + cat_feats
    if not feats:
        raise SystemExit("[ERROR] No hay features disponibles para entrenar.")

    X = df[feats].copy()
    y = df["label"].astype(int).values
    groups = df["run_id"].astype(str).values

    if net_feats:
        X[net_feats] = X[net_feats].fillna(X[net_feats].mean())
    for c in cat_feats:
        X[c] = X[c].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx]
    X_te, y_te = X.iloc[te_idx].copy(), y[te_idx]

    model = fit_model(X_tr, y_tr, X_te, y_te, cat_feats, args)
    p_raw_test = model.predict_proba(X_te)[:, 1].astype(float)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw_test, y_te)
    p_cal_test = iso.predict(p_raw_test)

    fi = pd.DataFrame({"feature": feats, "importance": model.get_feature_importance()})
    fi = fi.sort_values("importance", ascending=False)
    fi_path = os.path.join(args.report_dir, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    # Predicciones para todo el dataset (salida operativa del pipeline)
    p_raw_all = model.predict_proba(X)[:, 1].astype(float)
    p_cal_all = iso.predict(p_raw_all).astype(float)

    meta_cols = [c for c in ["run_id", "period", "vehID", "cell_id", "t_ref"] if c in df.columns]
    pred_base = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    raw_out = pred_base.copy()
    cal_out = pred_base.copy()
    raw_out["rho_hat"] = p_raw_all
    cal_out["rho_hat"] = p_cal_all
    raw_out.to_parquet(args.raw_out, index=False)
    cal_out.to_parquet(args.cal_out, index=False)

    # Ablation mínima real (sin red), para G7 sin valores hardcodeados
    roc_auc_no_network = None
    if net_feats:
        num_feats_nr, _, cat_feats_nr = build_feature_set(df, include_network=False)
        feats_nr = num_feats_nr + cat_feats_nr
        X_nr = df[feats_nr].copy()
        for c in cat_feats_nr:
            X_nr[c] = X_nr[c].astype(str)
        X_nr_tr = X_nr.iloc[tr_idx].copy()
        X_nr_te = X_nr.iloc[te_idx].copy()
        model_nr = fit_model(X_nr_tr, y_tr, X_nr_te, y_te, cat_feats_nr, args)
        p_nr = model_nr.predict_proba(X_nr_te)[:, 1].astype(float)
        roc_auc_no_network = float(roc_auc_score(y_te, p_nr))

    ablation_rows = [
        {"scenario": "Modelo Completo", "auc": float(roc_auc_score(y_te, p_cal_test))},
    ]
    if roc_auc_no_network is not None:
        ablation_rows.append({"scenario": "Sin Red", "auc": roc_auc_no_network})
    ablation_path = os.path.join(args.report_dir, "ablation_auc.csv")
    pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)

    out = {
        "model": "catboost_gbdt + isotonic",
        "with_network": True,
        "seed": args.seed,
        "n_total": int(len(df)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "features_num": num_feats,
        "features_net": net_feats,
        "features_cat": cat_feats,
        "roc_auc_raw": float(roc_auc_score(y_te, p_raw_test)),
        "ap_raw": float(average_precision_score(y_te, p_raw_test)),
        "brier_raw": float(brier_score_loss(y_te, p_raw_test)),
        "ece_raw": ece_score(y_te, p_raw_test, n_bins=10),
        "roc_auc_cal": float(roc_auc_score(y_te, p_cal_test)),
        "ap_cal": float(average_precision_score(y_te, p_cal_test)),
        "brier_cal": float(brier_score_loss(y_te, p_cal_test)),
        "ece_cal": ece_score(y_te, p_cal_test, n_bins=10),
        "roc_auc_cal_no_network": roc_auc_no_network,
    }

    model_path = os.path.join(args.out_dir, "catboost_gbdt.cbm")
    iso_path = os.path.join(args.out_dir, "isotonic.joblib")
    rep_path = os.path.join(args.report_dir, "report_catboost_isotonic.json")

    model.save_model(model_path)
    dump(iso, iso_path)

    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[OK] saved model:", model_path)
    print("[OK] saved iso  :", iso_path)
    print("[OK] saved report:", rep_path)
    print("[OK] saved raw rho_hat:", args.raw_out)
    print("[OK] saved calibrated rho_hat:", args.cal_out)
    print("[OK] saved ablation:", ablation_path)
    print("AUC raw:", out["roc_auc_raw"], "AUC cal:", out["roc_auc_cal"])
    print("ECE raw:", out["ece_raw"], "ECE cal:", out["ece_cal"])


if __name__ == "__main__":
    main()
