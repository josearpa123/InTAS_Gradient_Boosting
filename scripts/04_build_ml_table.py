#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd

FEATURES = [
    "speed_mean","speed_std",
    "accel_mean","accel_std",
    "jerk_abs_mean",
    "stops_count",
]
META = ["run_id","vehID","t_ref","cell_id","period","policy","rep","samples_window"]
TARGET = "label"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/dataset_windows.parquet")
    ap.add_argument("--out", default="data/ml_table.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.infile)

    # Asegurar columnas (incluyendo dinámicamente network_ y prb_usage_)
    net_cols = [c for c in df.columns if c.startswith("network_") or c.startswith("prb_usage_")]
    keep = [c for c in (META + FEATURES + [TARGET]) if c in df.columns] + net_cols
    df = df[keep].copy()

    # Imputación: jerk puede venir NaN en algunas ventanas -> poner 0 (interpretación: cambio brusco no observable)
    if "jerk_abs_mean" in df.columns:
        df["jerk_abs_mean"] = df["jerk_abs_mean"].fillna(0.0)

    # stops_count a numérico por si vino raro
    if "stops_count" in df.columns:
        df["stops_count"] = pd.to_numeric(df["stops_count"], errors="coerce").fillna(0.0)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print("[OK] ML table ->", args.out, "shape=", df.shape)
    print("label rate:", float(df[TARGET].mean()))
    if "jerk_abs_mean" in df.columns:
        print("jerk_abs_mean null%:", float(df["jerk_abs_mean"].isna().mean()))

if __name__ == "__main__":
    main()
