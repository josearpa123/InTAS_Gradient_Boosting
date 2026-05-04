#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, gzip, argparse, json
import pandas as pd
import xml.etree.ElementTree as ET

NUMERIC_WHITELIST = {
    "begin","end","speed","density","flow","occupancy","traveltime","haltings",
    "t","x","y","accel","angle","pos","duration","routeLength","waitingTime","timeLoss",
    "depart","arrival","departDelay"
}

def to_num_safe(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s

def gz_read(path: str) -> bytes:
    with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as f:
        return f.read()

def write_parquet(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path

def parse_run_id(run_id: str):
    """
    Espera: <period>__<policy>__repXX
    Devuelve: period, policy, rep(int)
    """
    m = re.match(r"^(?P<period>.+?)__(?P<policy>.+?)__rep(?P<rep>\d+)$", run_id)
    if not m:
        raise ValueError(f"run_id inválido: {run_id}. Formato esperado: <period>__<policy>__repXX")
    return m.group("period"), m.group("policy"), int(m.group("rep"))

def vehroute_to_parquet(xml_path, out_path, meta):
    # <vehicle><route edges="..." exitTimes="..."/></vehicle>
    src = xml_path if not xml_path.endswith(".gz") else io.BytesIO(gz_read(xml_path))
    it = ET.iterparse(src, events=("start","end"))
    _, root = next(it)
    rows=[]
    for ev, el in it:
        if ev=="end" and el.tag=="vehicle":
            vid = el.get("id")
            depart = float(el.get("depart","0"))
            rt = el.find("route")
            if rt is not None:
                edges = rt.get("edges","").split()
                exits = [float(x) for x in rt.get("exitTimes","").split() if x.strip()!=""]
                for i,e in enumerate(edges):
                    t_enter = depart if i==0 else (exits[i-1] if i-1<len(exits) else None)
                    t_exit  = exits[i] if i<len(exits) else None
                    rows.append({**meta, "vehID": vid, "edge_id": e, "idx_on_route": i,
                                 "t_enter": t_enter, "t_exit": t_exit})
            el.clear(); root.clear()
    df = pd.DataFrame(rows)
    if len(df)==0:
        df = pd.DataFrame(columns=list(meta.keys())+["vehID","edge_id","idx_on_route","t_enter","t_exit"])
    else:
        df["vehID"]=df["vehID"].astype("string")
        df["edge_id"]=df["edge_id"].astype("string")
        df["idx_on_route"]=df["idx_on_route"].astype("int32")
        df["t_enter"]=to_num_safe(df["t_enter"]).astype("float64")
        df["t_exit"]=to_num_safe(df["t_exit"]).astype("float64")
    write_parquet(df, out_path)

def fcd_to_parquet(xml_path, out_path, meta):
    # <timestep time="..."><vehicle id="..." x="..." y="..." speed="..." accel="..." edge="..." lane="..."/></timestep>
    src = xml_path if not xml_path.endswith(".gz") else io.BytesIO(gz_read(xml_path))
    it = ET.iterparse(src, events=("start","end"))
    _, root = next(it)
    rows=[]; cur_t=None
    for ev, el in it:
        if ev=="start" and el.tag=="timestep":
            cur_t = float(el.get("time","0"))
        if ev=="end" and el.tag=="vehicle":
            rows.append({**meta,
                "t": cur_t,
                "vehID": el.get("id"),
                "x": el.get("x"), "y": el.get("y"),
                "speed": el.get("speed"), "accel": el.get("accel"),
                "angle": el.get("angle"),
                "edge_id": el.get("edge"), "lane_id": el.get("lane")
            })
            el.clear(); root.clear()
    df = pd.DataFrame(rows)
    if len(df)==0:
        df = pd.DataFrame(columns=list(meta.keys())+["t","vehID","x","y","speed","accel","angle","edge_id","lane_id"])
    else:
        df["vehID"]=df["vehID"].astype("string")
        for c in ["t","x","y","speed","accel","angle"]:
            df[c]=to_num_safe(df[c]).astype("float32")
        for c in ["edge_id","lane_id"]:
            df[c]=df[c].astype("string")
    write_parquet(df, out_path)

def intervals_to_parquet(xml_path, out_path, meta, inner="edge", idkey="edge_id"):
    buf = gz_read(xml_path)
    root = ET.fromstring(buf.decode("utf-8", errors="ignore"))

    rows=[]
    for itv in root.findall(".//interval"):
        b = itv.get("begin"); e = itv.get("end")
        for el in itv.findall(f"./{inner}"):
            row = {**meta, "begin": b, "end": e}
            row.update(el.attrib)
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df)==0:
        df = pd.DataFrame(columns=list(meta.keys())+["begin","end",idkey])
        return write_parquet(df, out_path)

    if "id" in df.columns:
        df.rename(columns={"id": idkey}, inplace=True)
    df[idkey]=df[idkey].astype("string")
    df["begin"]=to_num_safe(df["begin"]).astype("float32")
    df["end"]=to_num_safe(df["end"]).astype("float32")

    for c in df.columns:
        if c in set(meta.keys()) | {"begin","end",idkey}:
            continue
        if c in NUMERIC_WHITELIST:
            df[c] = to_num_safe(df[c])

    write_parquet(df, out_path)

def lanechanges_to_parquet(xml_path, out_path, meta):
    buf = gz_read(xml_path)
    it = ET.iterparse(io.BytesIO(buf), events=("start","end"))
    _, root = next(it)
    rows=[]
    for ev, el in it:
        if ev=="end" and el.tag in ("change","laneChange","lanechange"):
            rows.append({**meta,
                "t": el.get("time"),
                "vehID": el.get("id") or el.get("veh") or el.get("vehicle"),
                "from_lane": el.get("from"),
                "to_lane": el.get("to"),
                "speed": el.get("speed"),
                "pos": el.get("pos"),
                "reason": el.get("reason")
            })
            el.clear(); root.clear()
    df=pd.DataFrame(rows)
    if len(df)==0:
        df=pd.DataFrame(columns=list(meta.keys())+["t","vehID","from_lane","to_lane","speed","pos","reason"])
    else:
        df["t"]=to_num_safe(df["t"]).astype("float32")
        for c in ["speed","pos"]:
            df[c]=to_num_safe(df[c]).astype("float32")
        for c in ["vehID","from_lane","to_lane","reason"]:
            df[c]=df[c].astype("string")
    write_parquet(df, out_path)

def tripinfo_to_parquet(xml_path, out_path, meta):
    buf = gz_read(xml_path)
    it = ET.iterparse(io.BytesIO(buf), events=("start","end"))
    _, root = next(it)
    rows=[]
    for ev, el in it:
        if ev=="end" and el.tag=="tripinfo":
            rows.append({**meta,
                "vehID": el.get("id"),
                "depart": el.get("depart"),
                "arrival": el.get("arrival"),
                "duration": el.get("duration"),
                "routeLength": el.get("routeLength"),
                "waitingTime": el.get("waitingTime"),
                "timeLoss": el.get("timeLoss"),
                "departDelay": el.get("departDelay"),
            })
            el.clear(); root.clear()
    df=pd.DataFrame(rows)
    if len(df)==0:
        df=pd.DataFrame(columns=list(meta.keys())+["vehID","depart","arrival","duration","routeLength","waitingTime","timeLoss","departDelay"])
    else:
        df["vehID"]=df["vehID"].astype("string")
        for c in ["depart","arrival","duration","routeLength","waitingTime","timeLoss","departDelay"]:
            df[c]=to_num_safe(df[c]).astype("float32")
    write_parquet(df, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="Ej: HC1_5pct__nearest__rep01")
    ap.add_argument("--runs_dir", default="sim/runs")
    ap.add_argument("--out", default="data/bronze")
    args = ap.parse_args()

    run_id=args.run_id
    period, policy, rep = parse_run_id(run_id)
    run_dir=os.path.join(args.runs_dir, run_id)

    if not os.path.isdir(run_dir):
        raise SystemExit(f"[ERROR] No existe run_dir: {run_dir}")

    meta={"run_id": run_id, "period": period, "policy": policy, "rep": rep}

    # nombres estándar que puso el runner
    files = {
        "vehroute":  ("vehroute.xml",        lambda src,out: vehroute_to_parquet(src,out,meta)),
        "fcd":       ("fcd.xml.gz",          lambda src,out: fcd_to_parquet(src,out,meta)),
        "edgedata":  ("edgedata.xml.gz",     lambda src,out: intervals_to_parquet(src,out,meta,inner="edge",idkey="edge_id")),
        "lanedata":  ("lanedata.xml.gz",     lambda src,out: intervals_to_parquet(src,out,meta,inner="lane",idkey="lane_id")),
        "lanechanges":("lanechanges.xml.gz", lambda src,out: lanechanges_to_parquet(src,out,meta)),
        "tripinfo":  ("tripinfo.xml.gz",     lambda src,out: tripinfo_to_parquet(src,out,meta)),
    }

    done={}
    for kind,(fname,fn) in files.items():
        src=os.path.join(run_dir,fname)
        if not os.path.isfile(src):
            print(f"[WARN] falta {kind}: {src}")
            continue
        outp=os.path.join(args.out, kind, f"{run_id}.parquet")
        fn(src,outp)
        done[kind]=outp

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", f"bronze_extract_{run_id}.json"),"w",encoding="utf-8") as f:
        json.dump({"run_id":run_id,"outputs":done}, f, indent=2)

    print("[OK] Bronze extraído:", run_id)
    for k,v in done.items():
        print("  ",k,"->",v)

if __name__=="__main__":
    main()
