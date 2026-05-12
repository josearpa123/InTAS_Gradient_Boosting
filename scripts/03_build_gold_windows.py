#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_label", default="data/silver/route_label.parquet")
    ap.add_argument("--route_events", default="data/silver/route_events.parquet")
    ap.add_argument("--fcd_dir", default="data/bronze/fcd")
    ap.add_argument("--fcd_cells_dir", default="data/silver/fcd_cells")
    ap.add_argument("--cell_exposure", default="data/silver/cell_exposure.parquet")
    ap.add_argument("--unified", default="data/unified_metrics.parquet")
    ap.add_argument("--out", default="data/dataset_windows.parquet")
    ap.add_argument("--window", type=float, default=30.0)   # segundos hacia atrás: [t_ref-window, t_ref]
    ap.add_argument("--step", type=float, default=1.0)      # timestep SUMO aproximado
    ap.add_argument("--neg_ratio", type=float, default=1.0) # negativos ~ positivos (1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Graceful fallback: if FCD bronze files don't exist but a pre-built
    # dataset_windows.parquet is already available, reuse it.
    fcd_dir = args.fcd_dir
    out_path = args.out
    if not os.path.isdir(fcd_dir) or not os.listdir(fcd_dir):
        if os.path.isfile(out_path):
            print(f"[WARN] FCD bronze directory '{fcd_dir}' is missing or empty.")
            print(f"[WARN] Reusing existing seed: {out_path}")
            return
        raise SystemExit(
            f"[ERROR] FCD bronze directory '{fcd_dir}' not found and no seed output exists. "
            "Run with INTAS_RUN_SUMO=1 to generate it."
        )

    rng = np.random.default_rng(args.seed)

    lab = pd.read_parquet(args.route_label)
    ev  = pd.read_parquet(args.route_events)

    # Normaliza nombres
    if "vehID" not in lab.columns: raise SystemExit("[ERROR] route_label sin vehID")
    if "vehID" not in ev.columns:  raise SystemExit("[ERROR] route_events sin vehID")

    # Solo runs presentes en los archivos
    runs = sorted(lab["run_id"].unique().tolist())

    # --- construir base de ejemplos positivos (uno por evento) ---
    pos = ev.merge(lab[["run_id","vehID","label"]], on=["run_id","vehID"], how="left")
    pos = pos[pos["label"]==1].copy()
    if pos.empty:
        raise SystemExit("[ERROR] No hay positivos en route_events/route_label.")

    # --- construir negativos: muestreo por run_id desde label=0, con t_ref muestreado del rango de ese run ---
    # Para que sea reproducible, usamos distribución de t_ref de positivos por run_id.
    pos_t_by_run = pos.groupby("run_id")["t_ref"].apply(list).to_dict()

    neg_pool = lab[lab["label"]==0].copy()
    neg_rows = []
    for run_id, g in neg_pool.groupby("run_id"):
        t_refs = pos_t_by_run.get(run_id, None)
        if not t_refs:
            continue
        # cuántos negativos para este run: proporcional a positivos del run
        n_pos = int((pos["run_id"]==run_id).sum())
        n_neg = int(np.ceil(n_pos * args.neg_ratio))
        # sample vehIDs sin reemplazo (si alcanza)
        vehs = g["vehID"].unique()
        if len(vehs)==0:
            continue
        pick = vehs if len(vehs) <= n_neg else rng.choice(vehs, size=n_neg, replace=False)
        # asignar t_ref: sample de t_refs positivos (con reemplazo)
        t_pick = rng.choice(np.array(t_refs, dtype=float), size=len(pick), replace=True)
        tmp = pd.DataFrame({"run_id": run_id, "vehID": pick, "t_ref": t_pick})
        neg_rows.append(tmp)
    neg = pd.concat(neg_rows, ignore_index=True) if neg_rows else pd.DataFrame(columns=["run_id","vehID","t_ref"])
    neg = neg.merge(lab[["run_id","vehID","label"]], on=["run_id","vehID"], how="left")
    neg = neg[neg["label"]==0].copy()

    base = pd.concat([pos[["run_id","vehID","t_ref","label"]], neg[["run_id","vehID","t_ref","label"]]], ignore_index=True)

    # --- Cargar exposure para asignar cell_id por ventana (celda con mayor time_s por veh en el run) ---
    cx = pd.read_parquet(args.cell_exposure)
    # para resolver cell_id por vehID-run: elegimos celda con mayor time_s
    cx_sorted = cx.sort_values(["run_id","vehID","time_s"], ascending=[True,True,False])
    cx_best = cx_sorted.drop_duplicates(["run_id","vehID"], keep="first")[["run_id","vehID","cell_id"]]

    base = base.merge(cx_best, on=["run_id","vehID"], how="left")

    # nos quedamos con ventanas que efectivamente tienen cell_id (vehículos que pisan celdas)
    base = base.dropna(subset=["cell_id"]).copy()
    base["cell_id"] = base["cell_id"].astype("string")

    # --- Cargar métricas de red (unified) para integración ---
    unified_path = args.unified
    net_feats = pd.DataFrame()
    if os.path.exists(unified_path):
        try:
            # Seleccionamos run_id, cell_id y columnas de red
            unified = pd.read_parquet(unified_path)
            net_cols = ["run_id", "cell_id"] + [c for c in unified.columns if c.startswith("network_") or c.startswith("prb_usage_")]
            net_feats = unified[net_cols].copy()
            net_feats["cell_id"] = net_feats["cell_id"].astype("string")
            # Agregamos por si hay duplicados inesperados
            net_feats = net_feats.groupby(["run_id", "cell_id"]).mean().reset_index()
            print(f"[INFO] Cargadas {len(net_feats.columns)-2} métricas de red para join.")
        except Exception as e:
            print(f"[WARN] Error cargando unified_metrics: {e}")

    # --- Construcción de features desde FCD ---
    rows=[]
    for run_id, g in base.groupby("run_id"):
        # ... (carga FCD existente)
        fcd_path = os.path.join(args.fcd_dir, f"{run_id}.parquet")
        if not os.path.isfile(fcd_path):
            raise SystemExit(f"[ERROR] falta FCD bronze: {fcd_path}")
        fcd = pd.read_parquet(fcd_path)

        # Normalizar columnas
        if "vehID" not in fcd.columns:
            if "veh_id" in fcd.columns: fcd = fcd.rename(columns={"veh_id":"vehID"})
        if "t" not in fcd.columns:
            raise SystemExit(f"[ERROR] FCD sin t en {run_id}")
        # numéricos
        for c in ["t","speed","accel"]:
            if c in fcd.columns:
                fcd[c]=pd.to_numeric(fcd[c], errors="coerce")
        fcd = fcd.dropna(subset=["vehID","t"]).copy()
        fcd["vehID"]=fcd["vehID"].astype("string")

        # jerk (derivada discreta de accel)
        if "accel" in fcd.columns:
            fcd = fcd.sort_values(["vehID","t"])
            fcd["jerk"] = fcd.groupby("vehID")["accel"].diff() / args.step
        else:
            fcd["jerk"] = np.nan

        # para acelerar, index por vehID
        for _, r in g.iterrows():
            veh = str(r["vehID"])
            t_ref = float(r["t_ref"])
            t0 = t_ref - args.window
            t1 = t_ref

            w = fcd[(fcd["vehID"]==veh) & (fcd["t"]>=t0) & (fcd["t"]<=t1)]
            if w.empty:
                continue

            speed = w["speed"] if "speed" in w.columns else pd.Series(dtype=float)
            accel = w["accel"] if "accel" in w.columns else pd.Series(dtype=float)
            jerk  = w["jerk"]  if "jerk"  in w.columns else pd.Series(dtype=float)

            # stops: speed ~ 0
            stops = int((speed.fillna(0) < 0.1).sum()) if len(speed) else 0

            row = {
                "run_id": run_id,
                "vehID": veh,
                "t_ref": t_ref,
                "cell_id": str(r["cell_id"]),
                "label": int(r["label"]),
                "period": lab.loc[lab["run_id"]==run_id, "period"].iloc[0] if "period" in lab.columns else None,
                "policy": lab.loc[lab["run_id"]==run_id, "policy"].iloc[0] if "policy" in lab.columns else None,
                "rep": int(lab.loc[lab["run_id"]==run_id, "rep"].iloc[0]) if "rep" in lab.columns else None,
                "samples_window": int(len(w)),
                "speed_mean": float(speed.mean()) if len(speed) else np.nan,
                "speed_std": float(speed.std(ddof=0)) if len(speed) else np.nan,
                "accel_mean": float(accel.mean()) if len(accel) else np.nan,
                "accel_std": float(accel.std(ddof=0)) if len(accel) else np.nan,
                "jerk_abs_mean": float(jerk.abs().mean()) if len(jerk) else np.nan,
                "stops_count": stops,
            }

            # Merge con red si existe
            if not net_feats.empty:
                match = net_feats[(net_feats["run_id"] == run_id) & (net_feats["cell_id"] == str(r["cell_id"]))]
                if not match.empty:
                    # Excluir run_id y cell_id para evitar duplicados en row
                    match_dict = match.iloc[0].drop(["run_id", "cell_id"]).to_dict()
                    row.update(match_dict)

            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("[ERROR] GOLD vacío. Puede que (vehID,t_ref) no tengan muestras en ventana o cell_id filtre demasiado.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)

    print("[OK] GOLD ->", args.out, out.shape)
    print("label rate:", float(out["label"].mean()))
    print(out.groupby("period")["label"].mean() if "period" in out.columns else out["label"].mean())
    print("cell_id unique:", int(out["cell_id"].nunique()))

if __name__=="__main__":
    main()
