#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def parse_cells_poly(path):
    root = ET.parse(path).getroot()
    polys = []
    for poly in root.findall(".//poly"):
        pid = poly.get("id")
        shape = poly.get("shape", "").strip()
        if not pid or not shape:
            continue
        pts = []
        for tok in shape.split():
            x_str, y_str = tok.split(",")
            pts.append((float(x_str), float(y_str)))
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        polys.append({
            "cell_id": pid,
            "xs": xs,
            "ys": ys,
            "bbox": (xmin, xmax, ymin, ymax)
        })
    if not polys:
        raise RuntimeError(f"No se encontraron <poly> en {path}")
    return polys

def points_in_poly_vectorized(x, y, xs, ys):
    """
    Ray casting vectorizado.
    x,y: arrays (N,)
    xs,ys: vértices del polígono (M,)
    return: mask bool (N,)
    """
    # Asegurar polígono cerrado
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs = np.r_[xs, xs[0]]
        ys = np.r_[ys, ys[0]]

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    inside = np.zeros(x.shape[0], dtype=bool)
    xj = xs[-1]
    yj = ys[-1]
    for i in range(len(xs)):
        xi = xs[i]; yi = ys[i]
        # cond: segmento cruza el rayo horizontal
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= intersect
        xj, yj = xi, yi
    return inside

def assign_cells(df, polys):
    # df debe tener x,y
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy()

    cell = np.array([None] * len(df), dtype=object)

    # Primero filtro por bbox (rápido) y luego point-in-poly
    for p in polys:
        xmin, xmax, ymin, ymax = p["bbox"]
        bbox_mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if not np.any(bbox_mask):
            continue
        # solo donde todavía no hay celda asignada
        pending = bbox_mask & (cell == None)
        if not np.any(pending):
            continue
        inside = points_in_poly_vectorized(x[pending], y[pending], p["xs"], p["ys"])
        idx = np.where(pending)[0]
        cell[idx[inside]] = p["cell_id"]

    df = df.copy()
    df["cell_id"] = pd.Series(cell, dtype="string")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--cells_poly", required=True)
    ap.add_argument("--bronze_dir", default="data/bronze/fcd")
    ap.add_argument("--out_dir", default="data/silver/fcd_cells")
    ap.add_argument("--chunksize", type=int, default=400000)  # ajustable
    args = ap.parse_args()

    in_path = os.path.join(args.bronze_dir, f"{args.run_id}.parquet")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"No existe FCD bronze: {in_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_id}.parquet")

    polys = parse_cells_poly(args.cells_poly)

    # leer y procesar en chunks para no explotar memoria
    df = pd.read_parquet(in_path)

    if "x" not in df.columns or "y" not in df.columns:
        raise RuntimeError(f"FCD no tiene columnas x/y en {in_path}. cols={list(df.columns)[:30]}")

    # si es pequeño, directo; si es grande, chunk manual por índices
    n = len(df)
    if n <= args.chunksize:
        out = assign_cells(df, polys)
    else:
        parts = []
        for start in range(0, n, args.chunksize):
            end = min(start + args.chunksize, n)
            chunk = df.iloc[start:end]
            parts.append(assign_cells(chunk, polys))
        out = pd.concat(parts, ignore_index=True)

    out.to_parquet(out_path, index=False)

    null_rate = float(out["cell_id"].isna().mean())
    uniq = int(out["cell_id"].dropna().nunique())
    print(f"[OK] fcd_cells: {args.run_id} -> {out_path}")
    print(f"     filas={len(out)} cell_id_null={null_rate:.4f} unique_cells={uniq}")

if __name__ == "__main__":
    main()
