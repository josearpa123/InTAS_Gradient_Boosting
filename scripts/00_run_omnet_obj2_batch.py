#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if p and Path(p).exists():
            return str(Path(p).resolve())
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OMNeT++ paired baseline/learned batch for Objective 2"
    )
    p.add_argument("--ini", default="scenarios/omnet/omnetpp.ini")
    p.add_argument("--scenario-root", default="scenarios/omnet")
    p.add_argument("--results-root", default="data/omnet_results")
    p.add_argument("--summary", default="reports/final/omnet_batch_summary.json")
    p.add_argument("--baseline-conf", default="SingleConnection-CBR-DL")
    p.add_argument("--learned-conf", default="DoubleConnection-CBR-DL")
    p.add_argument(
        "--sumo-trace-dir",
        default="data/sumo_traces_omnet",
        help=(
            "Directorio con trazas SUMO→OMNeT++ generadas por 00d. "
            "Si existe y contiene trazas, las corridas usan TraceFileMobility "
            "(SingleConnection-SUMO-DL / DoubleConnection-SUMO-DL) en lugar de "
            "LinearMobility. Si no existe, se usan los configs CBR estándar."
        ),
    )
    p.add_argument(
        "--injection-plan",
        default="",
        help=(
            "Ruta al JSON generado por 05b_inject_rho_omnet.py. "
            "Si se provee, la configuración 'learned' de cada corrida se "
            "selecciona según el plan (ρ̂ ≥ umbral → DoubleConnection, "
            "ρ̂ < umbral → SingleConnection). Sin este argumento se usa "
            "--learned-conf para todas las corridas."
        ),
    )
    p.add_argument("--rep-from", type=int, default=0)
    p.add_argument("--rep-to", type=int, default=14)
    p.add_argument("--opp-run", default="opp_run")
    p.add_argument("--periods", nargs="+", default=["HC1", "HC2", "HC3"])
    p.add_argument("--vfhs", nargs="+", default=["5", "10"])
    p.add_argument("--policies", nargs="+", default=["nearest", "ceiling"])
    p.add_argument("--timeout-sec", type=int, default=180)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--simu5g-lib", default=os.environ.get("SIMU5G_LIB", ""))
    p.add_argument("--inet-lib", default=os.environ.get("INET_LIB", ""))
    p.add_argument("--inet-src", default=os.environ.get("INET_SRC", ""))
    p.add_argument("--simu5g-src", default=os.environ.get("SIMU5G_SRC", ""))
    return p.parse_args()


def detect_sumo_trace(trace_dir: str, run_id: str) -> str | None:
    """Return path to first vehicle trace file for this run_id, or None."""
    if not trace_dir:
        return None
    run_path = Path(trace_dir) / run_id
    if not run_path.exists():
        return None
    traces = sorted(run_path.glob("*.trace"))
    if traces:
        return str(traces[0].resolve())
    return None


def sumo_mobility_overrides(trace_file: str) -> list[str]:
    """Return -G args that switch the UE to TraceFileMobility for this run."""
    return [
        "-G", f'*.ue[*].mobility.typename="TraceFileMobility"',
        "-G", f'*.ue[*].mobility.traceFile="{trace_file}"',
        "-G", '*.ue[*].mobility.updateInterval=0.1s',
    ]


def load_injection_plan(path: str) -> dict[str, str]:
    """Load run_id → conf_name mapping from injection plan JSON."""
    if not path:
        return {}
    plan_path = Path(path)
    if not plan_path.exists():
        print(f"[WARN] Injection plan not found: {path} — using default --learned-conf")
        return {}
    with plan_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    plan: dict[str, str] = payload.get("plan", {})
    print(
        f"[INFO] Injection plan loaded: {len(plan)} runs, "
        f"threshold={payload.get('threshold', '?')}, "
        f"learned={payload.get('n_learned', '?')}, "
        f"baseline={payload.get('n_baseline', '?')}"
    )
    return plan


def guess_paths(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    libs = []
    ned_paths = [
        str(Path(args.scenario_root).resolve()),
        str((Path(args.scenario_root) / "intas").resolve()),
    ]

    guessed_simu5g_lib = first_existing(
        [
            args.simu5g_lib,
            "/home/thunderbolt/ws/omnet/simu5g/src/libsimu5g.so",
            "/home/thunderbolt/work/omnet/simu5g/src/libsimu5g.so",
        ]
    )
    guessed_inet_lib = first_existing(
        [
            args.inet_lib,
            "/home/thunderbolt/ws/omnet/inet-4.5.4/src/libINET.so",
            "/home/thunderbolt/work/omnet/inet/src/libINET.so",
        ]
    )
    guessed_simu5g_src = first_existing(
        [
            args.simu5g_src,
            "/home/thunderbolt/ws/omnet/simu5g/src",
        ]
    )
    guessed_inet_src = first_existing(
        [
            args.inet_src,
            "/home/thunderbolt/ws/omnet/inet-4.5.4/src",
        ]
    )

    if guessed_simu5g_lib:
        libs.append(guessed_simu5g_lib)
    if guessed_inet_lib:
        libs.append(guessed_inet_lib)

    if guessed_inet_src:
        ned_paths.append(guessed_inet_src)
    if guessed_simu5g_src:
        simu5g_root = str(Path(guessed_simu5g_src).parents[0])
        ned_paths.extend(
            [
                guessed_simu5g_src,
                str(Path(simu5g_root) / "simulations"),
                str(Path(simu5g_root) / "simulations/nr"),
                str(Path(simu5g_root) / "simulations/nr/networks"),
            ]
        )

    # keep order, remove duplicates
    unique = []
    seen = set()
    for p in ned_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return libs, unique


def collect_new_outputs(root: Path, start_ts: float) -> dict[str, Path | None]:
    out = {"sca": None, "vec": None, "vci": None, "elog": None}
    for ext in out.keys():
        candidates = [
            p
            for p in root.rglob(f"*.{ext}")
            if p.is_file() and p.stat().st_mtime >= start_ts
        ]
        if candidates:
            out[ext] = max(candidates, key=lambda p: p.stat().st_mtime)
    return out


def run_one(
    args: argparse.Namespace,
    ini_path: Path,
    scenario_root: Path,
    out_root: Path,
    conf_name: str,
    rep: int,
    run_id: str,
    libs: list[str],
    ned_paths: list[str],
    extra_args: list[str] | None = None,
) -> tuple[bool, str]:
    out_dir = out_root / conf_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    target_sca = out_dir / "results.sca"
    target_vec = out_dir / "results.vec"
    target_vci = out_dir / "results.vci"
    target_elog = out_dir / "results.elog"
    if args.skip_existing and target_sca.exists():
        return True, "skip-existing"

    cmd = [
        args.opp_run,
        "-u",
        "Cmdenv",
        "-f",
        str(ini_path),
        "-c",
        conf_name,
        "-r",
        str(rep),
        "-n",
        ":".join(ned_paths),
    ]
    for lib in libs:
        cmd.extend(["-l", lib])
    if extra_args:
        cmd.extend(extra_args)

    log_path = out_dir / "opp_run.log"
    start_ts = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=scenario_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
        )

    produced = collect_new_outputs(scenario_root, start_ts)
    if produced["sca"] is not None:
        shutil.copy2(produced["sca"], target_sca)
    if produced["vec"] is not None:
        shutil.copy2(produced["vec"], target_vec)
    if produced["vci"] is not None:
        shutil.copy2(produced["vci"], target_vci)
    if produced["elog"] is not None:
        shutil.copy2(produced["elog"], target_elog)

    ok = proc.returncode == 0 and target_sca.exists()
    reason = "ok" if ok else f"failed(rc={proc.returncode})"
    return ok, reason


def main() -> int:
    args = parse_args()
    ini_path = Path(args.ini).resolve()
    scenario_root = Path(args.scenario_root).resolve()
    out_root = Path(args.results_root).resolve()
    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    if not ini_path.exists():
        raise SystemExit(f"[ERROR] Missing INI: {ini_path}")
    if not scenario_root.exists():
        raise SystemExit(f"[ERROR] Missing scenario root: {scenario_root}")

    libs, ned_paths = guess_paths(args)
    if not libs:
        raise SystemExit(
            "[ERROR] No OMNeT model libraries found (SIMU5G/INET). "
            "Set SIMU5G_LIB/INET_LIB or install compiled libs."
        )

    # Detección de trazas SUMO: si existen, se usa TraceFileMobility (acoplamiento
    # secuencial SUMO→OMNeT++) en lugar de LinearMobility.
    sumo_trace_dir = args.sumo_trace_dir
    sumo_traces_available = Path(sumo_trace_dir).exists() if sumo_trace_dir else False
    if sumo_traces_available:
        print(f"[INFO] Trazas SUMO detectadas en {sumo_trace_dir}")
        print("[INFO] Modo co-simulación secuencial: usando TraceFileMobility")
        # Configs que usan TraceFileMobility en lugar de LinearMobility
        effective_baseline = args.baseline_conf.replace("-CBR-", "-SUMO-")
        effective_learned_default = args.learned_conf.replace("-CBR-", "-SUMO-")
    else:
        print(f"[INFO] Sin trazas SUMO — usando LinearMobility (modo fallback)")
        effective_baseline = args.baseline_conf
        effective_learned_default = args.learned_conf

    # Plan de inyección de ρ̂: si se provee, selecciona la configuración
    # 'learned' de cada corrida según el estimador ML (05b_inject_rho_omnet.py).
    injection_plan = load_injection_plan(args.injection_plan)
    injection_mode = bool(injection_plan)
    if injection_mode:
        print("[INFO] Modo inyección ρ̂ activo: configuración 'learned' según plan ML")
    else:
        print(f"[INFO] Baseline={effective_baseline}, Learned={effective_learned_default}")

    baseline_dir = out_root / "baseline"
    learned_dir = out_root / "learned"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    learned_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total = 0
    ok_count = 0

    for period in args.periods:
        for vfh in args.vfhs:
            for policy in args.policies:
                for rep in range(args.rep_from, args.rep_to + 1):
                    total += 1
                    run_id = f"{period}_{vfh}pct__{policy}__rep{rep:02d}"
                    print(f"[OMNET] {total} run_id={run_id}")

                    # Trazas SUMO para esta corrida (integración secuencial)
                    trace_file = detect_sumo_trace(sumo_trace_dir, run_id) if sumo_traces_available else None
                    mobility_overrides = sumo_mobility_overrides(trace_file) if trace_file else []
                    if trace_file:
                        print(f"  [TRACE] {Path(trace_file).name}")
                    baseline_conf = effective_baseline
                    if injection_mode:
                        base_learned = injection_plan.get(run_id, effective_learned_default)
                        # Si el plan devolvió un conf CBR y hay trazas, migrar a SUMO equivalente
                        if sumo_traces_available and trace_file and "-CBR-" in base_learned:
                            learned_conf = base_learned.replace("-CBR-", "-SUMO-")
                        else:
                            learned_conf = base_learned
                    else:
                        learned_conf = effective_learned_default

                    b_ok, b_reason = run_one(
                        args,
                        ini_path=ini_path,
                        scenario_root=scenario_root,
                        out_root=baseline_dir,
                        conf_name=baseline_conf,
                        rep=rep,
                        run_id=run_id,
                        libs=libs,
                        ned_paths=ned_paths,
                        extra_args=mobility_overrides,
                    )
                    l_ok, l_reason = run_one(
                        args,
                        ini_path=ini_path,
                        scenario_root=scenario_root,
                        out_root=learned_dir,
                        conf_name=learned_conf,
                        rep=rep,
                        run_id=run_id,
                        libs=libs,
                        ned_paths=ned_paths,
                        extra_args=mobility_overrides,
                    )
                    pair_ok = b_ok and l_ok
                    if pair_ok:
                        ok_count += 1
                    rows.append(
                        {
                            "run_id": run_id,
                            "period": period,
                            "vfh": int(vfh),
                            "policy": policy,
                            "rep": rep,
                            "baseline_conf": baseline_conf,
                            "learned_conf": learned_conf,
                            "injection_mode": injection_mode,
                            "sumo_trace_used": trace_file is not None,
                            "sumo_trace_file": trace_file or "",
                            "baseline_ok": b_ok,
                            "baseline_reason": b_reason,
                            "learned_ok": l_ok,
                            "learned_reason": l_reason,
                            "pair_ok": pair_ok,
                        }
                    )

    summary = {
        "ini": str(ini_path),
        "scenario_root": str(scenario_root),
        "results_root": str(out_root),
        "baseline_conf": effective_baseline,
        "learned_conf": effective_learned_default,
        "sumo_traces_active": sumo_traces_available,
        "sumo_trace_dir": sumo_trace_dir if sumo_traces_available else None,
        "injection_mode": injection_mode,
        "libraries": libs,
        "ned_path": ned_paths,
        "total_pairs": total,
        "ok_pairs": ok_count,
        "fail_pairs": total - ok_count,
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("OMNeT Objective-2 batch finished")
    print(f"total pairs: {total}")
    print(f"ok pairs   : {ok_count}")
    print(f"fail pairs : {total - ok_count}")
    print(f"summary    : {summary_path}")
    print("=" * 72)
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
