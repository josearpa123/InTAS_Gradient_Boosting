#!/usr/bin/env python3
import os, json, glob, hashlib
from datetime import datetime

RUNS_DIR = "sim/runs"
OUT = "manifests/manifest_runs.jsonl"

# archivos SUMO esperados por corrida (según run_sumo_v2.py)
EXPECTED = [
    "fcd.xml.gz",
    "tripinfo.xml.gz",
    "vehroute.xml",
    "edgedata.xml.gz",
    "lanedata.xml.gz",
    "lanechanges.xml.gz",
    "manifest.json",
]

def sha1_file(path, block=1024*1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    runs = sorted([d for d in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d)])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    n_ok = 0
    n_missing = 0

    with open(OUT, "w", encoding="utf-8") as w:
        for d in runs:
            run_id = os.path.basename(d)
            mpath = os.path.join(d, "manifest.json")
            if not os.path.exists(mpath):
                rec = {"run_id": run_id, "status": "missing_manifest_json", "run_dir": os.path.abspath(d)}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_missing += 1
                continue

            meta = json.load(open(mpath, "r", encoding="utf-8"))
            files = {}
            missing = []

            for fn in EXPECTED:
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    # sha1 solo para manifest.json (barato) y vehroute.xml (moderado)
                    # (fcd/tripinfo/etc pueden ser muy grandes; se deja solo size)
                    do_hash = fn in ("manifest.json", "vehroute.xml")
                    files[fn] = {
                        "path": os.path.abspath(p),
                        "size_bytes": os.path.getsize(p),
                        "sha1": sha1_file(p) if do_hash else None,
                    }
                else:
                    missing.append(fn)

            rec = {
                "run_id": run_id,
                "created_at": meta.get("created_at"),
                "period": meta.get("period"),
                "policy": meta.get("policy"),
                "rep": meta.get("rep"),
                "seed_sumo": meta.get("seed_sumo"),
                "sumocfg": meta.get("sumocfg"),
                "outdir": meta.get("outdir"),
                "files": files,
                "missing_files": missing,
                "status": "ok" if not missing else "missing_outputs",
                "built_at": datetime.now().isoformat(),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print("Wrote:", OUT)
    print("runs:", len(runs), "ok_records:", n_ok, "missing_manifest:", n_missing)

if __name__ == "__main__":
    main()
