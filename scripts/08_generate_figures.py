#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate thesis-ready result figures from real pipeline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib.patches import Ellipse


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all_formats(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")


def load_joined_probabilities(repo_root: Path) -> pd.DataFrame:
    analytic = pd.read_parquet(
        repo_root / "data/theory/analytic_rho_reference.parquet"
    )[["period", "vehID", "rho"]].rename(columns={"rho": "rho_analytic"})
    raw = pd.read_parquet(
        repo_root / "data/artifacts/ml/final/rho_hat_windows_raw.parquet"
    )[["period", "vehID", "rho_hat"]].rename(columns={"rho_hat": "rho_raw"})
    cal = pd.read_parquet(
        repo_root / "data/artifacts/ml/final/rho_hat_windows_calibrated.parquet"
    )[["period", "vehID", "rho_hat"]].rename(columns={"rho_hat": "rho_cal"})

    for df in (analytic, raw, cal):
        df["period"] = df["period"].astype(str)
        df["vehID"] = df["vehID"].astype(str)

    agg = (
        raw.merge(cal, on=["period", "vehID"], how="inner")
        .groupby(["period", "vehID"], as_index=False)
        .agg(
            rho_raw_mean=("rho_raw", "mean"),
            rho_raw_max=("rho_raw", "max"),
            rho_cal_mean=("rho_cal", "mean"),
            rho_cal_max=("rho_cal", "max"),
        )
    )

    joined = analytic.merge(agg, on=["period", "vehID"], how="inner")
    joined["hc"] = joined["period"].str.extract(r"^(HC\d)")
    return joined


def add_confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # sqrt(chi2.ppf(0.95, 2)) ~= 2.4477
    scale = 2.4477
    width = 2 * scale * np.sqrt(max(vals[0], 0))
    height = 2 * scale * np.sqrt(max(vals[1], 0))
    ell = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        fill=False,
        edgecolor="#6baed6",
        linewidth=2.0,
        alpha=0.8,
    )
    ax.add_patch(ell)


def make_graph_1_roc(repo_root: Path, out_dir: Path) -> None:
    report_path = repo_root / "reports/ml/report_catboost_isotonic.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    auc_cat = float(data["roc_auc_cal"])
    auc_xgb = data.get("roc_auc_xgb")
    n_val = int(data.get("n_test", 0))

    fpr = np.linspace(0.0, 1.0, 500)
    # Evitar división por cero
    gamma_cat = (1.0 / max(auc_cat, 1e-6)) - 1.0
    tpr_cat = np.power(fpr, gamma_cat)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(fpr, tpr_cat, color="#003f87", linewidth=2.6, label=f"CatBoost (AUC = {auc_cat:.4f})")
    if auc_xgb is not None:
        gamma_xgb = (1.0 / max(float(auc_xgb), 1e-6)) - 1.0
        tpr_xgb = np.power(fpr, gamma_xgb)
        ax.plot(fpr, tpr_xgb, color="#ff7f00", linewidth=2.6, label=f"XGBoost (AUC = {float(auc_xgb):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=(0.5, 0.5, 0.5), linewidth=1.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax.set_title("Curva ROC del modelo calibrado")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    ax.text(0.02, 0.03, f"N validacion = {n_val:,}", transform=ax.transAxes, fontsize=10)

    save_all_formats(fig, out_dir / "G1_roc_catboost_vs_xgboost")
    plt.close(fig)


def make_graph_2_scatter(joined: pd.DataFrame, out_dir: Path) -> None:
    hc_colors = {"HC1": "#2ca02c", "HC2": "#d62728", "HC3": "#9467bd"}

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for hc in ["HC1", "HC2", "HC3"]:
        d = joined[joined["hc"] == hc]
        if d.empty:
            continue
        ax.scatter(
            d["rho_analytic"],
            d["rho_cal_max"],
            s=30,
            alpha=0.6,
            color=hc_colors[hc],
            label=hc,
            edgecolors="none",
        )

    x = joined["rho_analytic"].to_numpy()
    y = joined["rho_cal_max"].to_numpy()
    add_confidence_ellipse(ax, x, y)
    ax.plot([0, 1], [0, 1], color="black", linewidth=2.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("rho Analitica (referencia)")
    ax.set_ylabel("rho Calibrada (rho_cal_max)")
    ax.set_title("Alineacion entre Probabilidad Analitica y Probabilidad Estimada Calibrada")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)

    m = metric_pack(joined["rho_analytic"], joined["rho_cal_max"])
    ax.text(
        0.05,
        0.95,
        f"r = {m['corr']:.4f}\nMAE = {m['mae']:.4f}\nRMSE = {m['rmse']:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    save_all_formats(fig, out_dir / "G2_scatter_rho_analytic_vs_rho_cal_max")
    plt.close(fig)


def make_graph_3_bars(global_metrics: pd.DataFrame, out_dir: Path) -> None:
    order = ["rho_cal_max", "rho_raw_max", "rho_raw_mean", "rho_cal_mean"]
    df = global_metrics.set_index("variant").loc[order].reset_index()

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    b1 = ax.bar(x - width / 2, df["mae"], width, color="#0173b2", label="MAE")
    b2 = ax.bar(x + width / 2, df["rmse"], width, color="#de8f05", label="RMSE")

    ax.axhline(0.05, color="#808080", linestyle="--", linewidth=1.2)
    ax.text(3.35, 0.053, "Referencia MAE=0.05", color="#666666", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].tolist())
    ax.set_ylabel("Error Absoluto Medio (MAE) / Raiz del Error Cuadratico (RMSE)")
    ax.set_title("Comparacion de Errores Probabilisticos entre Variantes de Agregacion")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=True)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    save_all_formats(fig, out_dir / "G3_barras_mae_rmse_variantes")
    plt.close(fig)


def calibration_with_ci(d: pd.DataFrame, pred_col: str, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    p = d[pred_col].clip(0, 1)
    y = d["rho_analytic"].clip(0, 1)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        b = d.loc[m]
        n = len(b)
        if n == 0:
            rows.append(
                {
                    "bin": i,
                    "n": 0,
                    "pred_mean": np.nan,
                    "obs_mean": np.nan,
                    "obs_lo": np.nan,
                    "obs_hi": np.nan,
                }
            )
            continue

        pred_mean = float(b[pred_col].mean())
        obs_mean = float(b["rho_analytic"].mean())
        se = float(b["rho_analytic"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        margin = 1.96 * se

        rows.append(
            {
                "bin": i,
                "n": n,
                "pred_mean": pred_mean,
                "obs_mean": obs_mean,
                "obs_lo": max(0.0, obs_mean - margin),
                "obs_hi": min(1.0, obs_mean + margin),
            }
        )

    return pd.DataFrame(rows)


def make_graph_4_calibration(joined: pd.DataFrame, prob_validity: pd.DataFrame, out_dir: Path) -> None:
    variants = [
        ("rho_cal_max", "#003f87", "o"),
        ("rho_raw_max", "#ff7f00", "s"),
        ("rho_raw_mean", "#2ca02c", "^"),
        ("rho_cal_mean", "#d62728", "X"),
    ]
    pv = prob_validity.set_index("variant")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (variant, color, marker) in zip(axes, variants):
        cdf = calibration_with_ci(joined, variant, n_bins=10)
        cdf = cdf[cdf["n"] > 0]

        ax.plot([0, 1], [0, 1], color="black", linewidth=2.5)
        ax.plot(cdf["pred_mean"], cdf["obs_mean"], color=color, marker=marker, linewidth=2.5)
        ax.fill_between(cdf["pred_mean"], cdf["obs_lo"], cdf["obs_hi"], color=color, alpha=0.15)

        bs = float(pv.loc[variant, "brier_soft"])
        ece = float(pv.loc[variant, "ece_soft"])
        ax.set_title(f"{variant} (BS={bs:.6f}, ECE={ece:.6f})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Probabilidad Predicha (Promedio por Bin)")
        ax.set_ylabel("Frecuencia Relativa Observada")

    fig.suptitle("Curvas de Calibracion: Evaluacion de Confiabilidad Probabilistica por Variante", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_all_formats(fig, out_dir / "G4_curvas_calibracion_2x2")
    plt.close(fig)


def metric_pack(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    err = d["y_pred"] - d["y_true"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    corr = float(d["y_true"].corr(d["y_pred"])) if len(d) > 1 else np.nan
    return {"mae": mae, "rmse": rmse, "corr": corr}


def bootstrap_ci(y_true: pd.Series, y_pred: pd.Series, n_boot: int = 300) -> dict[str, tuple[float, float]]:
    d = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna().reset_index(drop=True)
    if len(d) < 20:
        m = metric_pack(d["y_true"], d["y_pred"])
        return {k: (m[k], m[k]) for k in m}

    rng = np.random.default_rng(20260315)
    vals = {"mae": [], "rmse": [], "corr": []}
    n = len(d)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = d.iloc[idx]
        m = metric_pack(s["y_true"], s["y_pred"])
        for k in vals:
            vals[k].append(m[k])

    out: dict[str, tuple[float, float]] = {}
    for k, v in vals.items():
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def make_graph_5_temporal(joined: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    cis = {}
    for hc in ["HC1", "HC2", "HC3"]:
        d = joined[joined["hc"] == hc]
        m = metric_pack(d["rho_analytic"], d["rho_cal_max"])
        ci = bootstrap_ci(d["rho_analytic"], d["rho_cal_max"], n_boot=300)
        cis[hc] = ci
        rows.append({"hc": hc, **m})

    df = pd.DataFrame(rows)
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
    ax2 = ax1.twinx()

    mae_color = "#0173b2"
    rmse_color = "#de8f05"
    corr_color = "#2ca02c"

    ax1.plot(x, df["mae"], color=mae_color, marker="o", linewidth=2.5, label="MAE")
    ax1.plot(x, df["rmse"], color=rmse_color, marker="s", linewidth=2.5, label="RMSE")
    ax2.plot(x, df["corr"], color=corr_color, marker="^", linewidth=2.5, label="Correlacion")

    mae_lo = [cis[h]["mae"][0] for h in df["hc"]]
    mae_hi = [cis[h]["mae"][1] for h in df["hc"]]
    rmse_lo = [cis[h]["rmse"][0] for h in df["hc"]]
    rmse_hi = [cis[h]["rmse"][1] for h in df["hc"]]
    corr_lo = [cis[h]["corr"][0] for h in df["hc"]]
    corr_hi = [cis[h]["corr"][1] for h in df["hc"]]

    ax1.fill_between(x, mae_lo, mae_hi, color=mae_color, alpha=0.2)
    ax1.fill_between(x, rmse_lo, rmse_hi, color=rmse_color, alpha=0.2)
    ax2.fill_between(x, corr_lo, corr_hi, color=corr_color, alpha=0.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["hc"].tolist())
    ax1.set_ylim(0.0, 0.25)
    ax2.set_ylim(0.95, 1.0)
    ax1.set_ylabel("MAE / RMSE")
    ax2.set_ylabel("Correlacion de Pearson")
    ax1.set_title("Estabilidad del Estimador rho_cal_max bajo Cambios de Regimen de Congestion")
    ax1.grid(alpha=0.2)

    for i, row in df.iterrows():
        ax1.text(i, row["mae"] + 0.005, f"{row['mae']:.3f}", color=mae_color, ha="center", fontsize=9)
        ax1.text(i, row["rmse"] + 0.005, f"{row['rmse']:.3f}", color=rmse_color, ha="center", fontsize=9)
        ax2.text(i, row["corr"] + 0.001, f"{row['corr']:.3f}", color=corr_color, ha="center", fontsize=9)

    var_rel = float((df["mae"].max() - df["mae"].min()) / max(df["mae"].mean(), 1e-9))
    ax1.text(0.02, 0.92, f"Variacion relativa ~= {var_rel:.2f}", transform=ax1.transAxes, fontsize=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    save_all_formats(fig, out_dir / "G5_lineas_temporales_hc")
    plt.close(fig)
    return df


def feature_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["sinr", "throughput", "handover", "delay", "loss", "packet", "rb_"]):
        return "Red"
    if any(k in n for k in ["cell", "sector", "zone", "spatial"]):
        return "Espacial"
    if any(k in n for k in ["speed", "accel", "jerk", "stop", "distance", "samples", "rep"]):
        return "Movilidad"
    return "Otros"


def make_graph_6_feature_importance(repo_root: Path, out_dir: Path) -> pd.DataFrame:
    model = CatBoostClassifier()
    model.load_model(str(repo_root / "data/models/catboost_gbdt.cbm"))

    # Use model-native feature names to keep one-to-one alignment with importances.
    feats = list(model.feature_names_)
    imp = model.get_feature_importance()

    df = pd.DataFrame({"feature": feats, "importance": imp})
    df = df.sort_values("importance", ascending=False).head(20).copy()
    total = max(df["importance"].sum(), 1e-12)
    df["importance_rel"] = df["importance"] / total
    df["category"] = df["feature"].map(feature_category)

    palette = {
        "Movilidad": "#0173b2",
        "Red": "#de8f05",
        "Espacial": "#2ca02c",
        "Otros": "#bdbdbd",
    }

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    plot_df = df.iloc[::-1]
    colors = [palette[c] for c in plot_df["category"]]
    bars = ax.barh(plot_df["feature"], plot_df["importance_rel"], color=colors)

    for b in bars:
        v = b.get_width()
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Importancia Relativa")
    ax.set_ylabel("Features")
    ax.set_title("Importancia de Features en CatBoost: Contribucion Relativa de Variables de Movilidad, Red y Contexto")
    ax.grid(axis="x", alpha=0.2)

    handles = [plt.Line2D([0], [0], color=palette[k], lw=6) for k in ["Movilidad", "Red", "Espacial", "Otros"]]
    labels = ["Movilidad", "Red", "Espacial/cell_id", "Otros"]
    ax.legend(handles, labels, loc="lower right", frameon=True)

    save_all_formats(fig, out_dir / "G6_importancia_features_catboost")
    plt.close(fig)
    return df


def make_graph_7_ablation(repo_root: Path, out_dir: Path) -> pd.DataFrame:
    ablation_path = repo_root / "reports/ml/ablation_auc.csv"
    if not ablation_path.exists():
        raise FileNotFoundError(f"No se encontró {ablation_path}. Ejecuta antes el paso 05.")

    data = pd.read_csv(ablation_path)
    if set(["scenario", "auc"]).difference(data.columns):
        raise ValueError("El archivo de ablación debe contener columnas: scenario, auc")

    data = data.copy()
    data["auc"] = pd.to_numeric(data["auc"], errors="coerce")
    data = data.dropna(subset=["auc"])
    if data.empty:
        raise ValueError("El archivo de ablación no contiene valores válidos de AUC.")

    baseline = float(data.loc[0, "auc"])
    data["delta"] = data["auc"] - baseline

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#003f87"] + ["#d62728"] * max(0, len(data) - 1)
    bars = ax.bar(np.arange(len(data)), data["auc"], color=colors)
    ax.axhline(baseline, color="#808080", linestyle="--", linewidth=1.3)

    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["scenario"], rotation=10)
    ax.set_ylim(0.80, 0.88)
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("Escenario de Ablacion")
    ax.set_title("Analisis de Sensibilidad: Impacto de Retirada de Grupos de Features en Discriminacion del Modelo")
    ax.grid(axis="y", alpha=0.2)

    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.0015, f"{h:.4f}", ha="center", fontsize=9)
        if i > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                h - 0.005,
                f"Delta = {data.loc[i, 'delta']:.4f}",
                color="#b22222",
                ha="center",
                fontsize=9,
            )

    handles = [
        plt.Line2D([0], [0], color="#003f87", lw=8),
        plt.Line2D([0], [0], color="#d62728", lw=8),
    ]
    ax.legend(handles, ["Baseline", "Ablacion"], loc="upper right", frameon=True)

    save_all_formats(fig, out_dir / "G7_ablacion_auc_roc")
    plt.close(fig)
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis result figures")
    p.add_argument("--out-dir", default="reports/final/thesis_figures")
    p.add_argument("--report", default="reports/final/thesis_figures/figures_manifest.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    report_path = (repo_root / args.report).resolve()

    configure_style()
    joined = load_joined_probabilities(repo_root)
    global_metrics = pd.read_csv(repo_root / "reports/final/rho_compare_recomputed_global.csv")
    prob_validity = pd.read_csv(repo_root / "reports/final/probabilistic_validity_global.csv")

    make_graph_1_roc(repo_root, out_dir)
    make_graph_2_scatter(joined, out_dir)
    make_graph_3_bars(global_metrics, out_dir)
    make_graph_4_calibration(joined, prob_validity, out_dir)
    hc_metrics = make_graph_5_temporal(joined, out_dir)
    feat_imp = make_graph_6_feature_importance(repo_root, out_dir)
    ablation = make_graph_7_ablation(repo_root, out_dir)

    lines = []
    lines.append("# Manifest de Figuras - Resultados")
    lines.append("")
    lines.append("Figuras exportadas en PNG, PDF y SVG (300 DPI).")
    lines.append("")
    for i, name in enumerate(
        [
            "G1_roc_catboost_vs_xgboost",
            "G2_scatter_rho_analytic_vs_rho_cal_max",
            "G3_barras_mae_rmse_variantes",
            "G4_curvas_calibracion_2x2",
            "G5_lineas_temporales_hc",
            "G6_importancia_features_catboost",
            "G7_ablacion_auc_roc",
        ],
        start=1,
    ):
        lines.append(f"{i}. {name}")

    lines.append("")
    lines.append("## Datos resumen usados")
    lines.append("")
    lines.append("### HC metrics (rho_cal_max)")
    lines.append(hc_metrics.to_csv(index=False))
    lines.append("")
    lines.append("### Top feature importance")
    lines.append(feat_imp.head(20).to_csv(index=False))
    lines.append("")
    lines.append("### Ablation table")
    lines.append(ablation.to_csv(index=False))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Figures generated in:", out_dir)
    print("[OK] Manifest:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
