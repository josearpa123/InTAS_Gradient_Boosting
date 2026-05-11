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
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


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


def fit_catboost(X_tr, y_tr, X_te, y_te, cat_feats, args):
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


def encode_categoricals_for_xgb(X, cat_feats, encoders=None):
    """Label-encode categorical columns for XGBoost (returns encoded copy + fitted encoders)."""
    X_enc = X.copy()
    fit_encoders = encoders is None
    if fit_encoders:
        encoders = {}
    for c in cat_feats:
        if fit_encoders:
            le = LabelEncoder()
            X_enc[c] = le.fit_transform(X_enc[c].astype(str))
            encoders[c] = le
        else:
            le = encoders[c]
            known = set(le.classes_)
            X_enc[c] = X_enc[c].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            X_enc[c] = le.transform(X_enc[c])
    return X_enc, encoders


def fit_xgboost(X_tr, y_tr, X_te, y_te, cat_feats, args):
    X_tr_enc, encoders = encode_categoricals_for_xgb(X_tr, cat_feats)
    X_te_enc, _ = encode_categoricals_for_xgb(X_te, cat_feats, encoders=encoders)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
        n_estimators=args.iterations,
        max_depth=args.depth,
        learning_rate=args.lr,
        reg_lambda=args.l2,
        early_stopping_rounds=args.early_stop,
        verbosity=1,
    )
    model.fit(
        X_tr_enc,
        y_tr,
        eval_set=[(X_te_enc, y_te)],
        verbose=200,
    )
    return model, encoders


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

    # ── CatBoost (modelo principal) ──────────────────────────────────────────
    catboost_model = fit_catboost(X_tr, y_tr, X_te, y_te, cat_feats, args)
    p_raw_test = catboost_model.predict_proba(X_te)[:, 1].astype(float)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw_test, y_te)
    p_cal_test = iso.predict(p_raw_test)

    # ── XGBoost (contraste robusto / referencia comparativa) ─────────────────
    xgb_model, xgb_encoders = fit_xgboost(X_tr, y_tr, X_te, y_te, cat_feats, args)
    X_te_enc, _ = encode_categoricals_for_xgb(X_te, cat_feats, encoders=xgb_encoders)
    p_xgb_test = xgb_model.predict_proba(X_te_enc)[:, 1].astype(float)

    xgb_auc = float(roc_auc_score(y_te, p_xgb_test))
    xgb_brier = float(brier_score_loss(y_te, p_xgb_test))
    xgb_ece = ece_score(y_te, p_xgb_test, n_bins=10)

    # ── Feature importance CatBoost ───────────────────────────────────────────
    fi = pd.DataFrame({"feature": feats, "importance": catboost_model.get_feature_importance()})
    fi = fi.sort_values("importance", ascending=False)
    fi_path = os.path.join(args.report_dir, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    # ── Predicciones para todo el dataset (salida operativa del pipeline) ─────
    p_raw_all = catboost_model.predict_proba(X)[:, 1].astype(float)
    p_cal_all = iso.predict(p_raw_all).astype(float)

    meta_cols = [c for c in ["run_id", "period", "vehID", "cell_id", "t_ref"] if c in df.columns]
    pred_base = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    raw_out = pred_base.copy()
    cal_out = pred_base.copy()
    raw_out["rho_hat"] = p_raw_all
    cal_out["rho_hat"] = p_cal_all
    raw_out.to_parquet(args.raw_out, index=False)
    cal_out.to_parquet(args.cal_out, index=False)

    # ── Ablation mínima real (sin red) ────────────────────────────────────────
    roc_auc_no_network = None
    if net_feats:
        num_feats_nr, _, cat_feats_nr = build_feature_set(df, include_network=False)
        feats_nr = num_feats_nr + cat_feats_nr
        X_nr = df[feats_nr].copy()
        for c in cat_feats_nr:
            X_nr[c] = X_nr[c].astype(str)
        X_nr_tr = X_nr.iloc[tr_idx].copy()
        X_nr_te = X_nr.iloc[te_idx].copy()
        model_nr = fit_catboost(X_nr_tr, y_tr, X_nr_te, y_te, cat_feats_nr, args)
        p_nr = model_nr.predict_proba(X_nr_te)[:, 1].astype(float)
        roc_auc_no_network = float(roc_auc_score(y_te, p_nr))

    ablation_rows = [
        {"scenario": "CatBoost Completo", "auc": float(roc_auc_score(y_te, p_cal_test))},
        {"scenario": "XGBoost (contraste)", "auc": xgb_auc},
    ]
    if roc_auc_no_network is not None:
        ablation_rows.append({"scenario": "CatBoost Sin Red", "auc": roc_auc_no_network})
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
        # roc_auc_xgb at top level — consumed by 08_generate_figures.py (G1 ROC curve)
        "roc_auc_xgb": xgb_auc,
        "xgboost_reference": {
            "model": "xgboost",
            "roc_auc": xgb_auc,
            "brier": xgb_brier,
            "ece": xgb_ece,
        },
    }

    model_path = os.path.join(args.out_dir, "catboost_gbdt.cbm")
    iso_path = os.path.join(args.out_dir, "isotonic.joblib")
    xgb_path = os.path.join(args.out_dir, "xgboost_ref.json")
    rep_path = os.path.join(args.report_dir, "report_catboost_isotonic.json")

    catboost_model.save_model(model_path)
    dump(iso, iso_path)
    xgb_model.save_model(xgb_path)
    dump(xgb_encoders, os.path.join(args.out_dir, "xgb_encoders.joblib"))

    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[OK] saved catboost:", model_path)
    print("[OK] saved xgboost :", xgb_path)
    print("[OK] saved iso     :", iso_path)
    print("[OK] saved report  :", rep_path)
    print("[OK] saved raw rho_hat:", args.raw_out)
    print("[OK] saved calibrated rho_hat:", args.cal_out)
    print("[OK] saved ablation:", ablation_path)
    print()
    print("=== CatBoost (principal) ===")
    print("AUC raw:", out["roc_auc_raw"], " AUC cal:", out["roc_auc_cal"])
    print("ECE raw:", out["ece_raw"],     " ECE cal:", out["ece_cal"])
    print()
    print("=== XGBoost (contraste robusto) ===")
    print("AUC:", xgb_auc, " Brier:", xgb_brier, " ECE:", xgb_ece)


if __name__ == "__main__":
    main()
