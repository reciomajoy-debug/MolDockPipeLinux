#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4e (Worker) — SHARD SAFE (Linux, single GPU)

- Runs docking on ONE GPU (via --gpu <id> → sets CUDA_VISIBLE_DEVICES)
- Writes ONLY a per-GPU shard manifest (state/manifest_gpu<N>.csv) atomically
- NEVER touches state/manifest.csv, results/summary.csv, results/leaderboard.csv
- Preserves SAFE_RESUME batching and atom-type validation
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Iterable

# ======================== Tunables ========================
BATCH_SIZE  = 64
SAFE_RESUME = True
KEEP_TMP    = False
# =========================================================

STOP_REQUESTED = False
HARD_STOP = False

def _sigint(_, __):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C — finishing current batch then exiting cleanly…", flush=True)
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will exit ASAP after harvest.", flush=True)
signal.signal(signal.SIGINT, _sigint)

BASE = Path(".").resolve()
DIR_PREP = BASE / "prepared_ligands"
DIR_RESULTS = BASE / "results"
DIR_STATE = BASE / "state"
DIR_STATE.mkdir(parents=True, exist_ok=True)
DIR_RESULTS.mkdir(parents=True, exist_ok=True)

FILE_MANIFEST_GLOBAL = DIR_STATE / "manifest.csv"  # NOT WRITTEN HERE

# ------------------------------ IO utils ------------------------------
def now_iso()->str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv(path: Path)->list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def atomic_write_csv(path: Path, rows: list[dict], headers: list[str])->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def sha1_of_file(p: Path)->str:
    h=hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]

def load_manifest(path: Path)->dict[str,dict]:
    out={}
    for r in read_csv(path):
        rid = r.get("id","")
        if not rid:
            continue
        row={k:r.get(k,"") for k in MANIFEST_FIELDS}
        out[rid] = row
    return out

def save_manifest(path: Path, manifest:dict[str,dict])->None:
    rows = [{k:v.get(k,"") for k in MANIFEST_FIELDS} for _,v in sorted(manifest.items())]
    atomic_write_csv(path, rows, MANIFEST_FIELDS)

# -------------------------- Atom-types check --------------------------
ALLOWED_AD4_TYPES = {
    "C","A","N","O","S","H","P","F","Cl","Br","I",
    "HD","NA","OA","SA",
    "Zn","Fe","Mg","Mn","Ca","Cu","Ni","Co","K","Na"
}

def get_pdbqt_atom_types(path: Path) -> set[str]:
    types=set()
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith(("ATOM","HETATM")):
                toks=line.split()
                if toks:
                    types.add(toks[-1])
    except Exception:
        pass
    return types

def pdbqt_has_only_allowed_types(path: Path) -> tuple[bool,str]:
    ts=get_pdbqt_atom_types(path)
    bad=[t for t in ts if t not in ALLOWED_AD4_TYPES]
    if bad:
        return False, "Unsupported AD4 atom types: "+",".join(sorted(set(bad)))
    return True, "OK"

# -------------------------- Binary + Config --------------------------
def find_vinagpu_binary()->Path:
    candidates = [
        BASE / "AutoDock-Vina-GPU-2-1",
        shutil.which("AutoDock-Vina-GPU-2-1") and Path(shutil.which("AutoDock-Vina-GPU-2-1")),
        BASE / "vina-gpu",
        BASE / "Vina-GPU",
    ]
    for p in candidates:
        if not p:
            continue
        p = Path(p)
        if p.exists():
            try:
                mode = p.stat().st_mode
                if not (mode & 0o111):
                    p.chmod(mode | 0o755)
            except Exception:
                pass
            return p.resolve()
    raise SystemExit("❌ AutoDock-Vina-GPU-2-1 binary not found in project root or PATH.")

def parse_cfg(path: Path)->Dict[str,str]:
    if not path.exists():
        raise SystemExit(f"❌ Config not found: {path}")
    conf={}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line=line.split("#",1)[0].strip()
        if "=" not in line:
            continue
        k,v=line.split("=",1)
        conf[k.strip().lower()]=v.strip()
    return conf

def as_float(d:Dict[str,str],k:str,default:float)->float:
    try:
        return float(d.get(k,default))
    except Exception:
        return float(default)

def as_int(d:Dict[str,str],k:str,default:int)->int:
    try:
        return int(str(d.get(k,default)).strip())
    except Exception:
        return int(default)

def load_runtime(vgpu: Path):
    cfg_gpu = vgpu.parent/"VinaGPUConfig.txt"
    cfg_cpu = vgpu.parent/"VinaConfig.txt"
    cfg_path = cfg_gpu if cfg_gpu.exists() else cfg_cpu
    conf = parse_cfg(cfg_path)

    box = {
        "center_x": as_float(conf,"center_x",0.0),
        "center_y": as_float(conf,"center_y",0.0),
        "center_z": as_float(conf,"center_z",0.0),
        "size_x":   as_float(conf,"size_x",20.0),
        "size_y":   as_float(conf,"size_y",20.0),
        "size_z":   as_float(conf,"size_z",20.0),
    }
    gcfg = {
        "thread": max(1000, as_int(conf,"thread",10000)),
        "search_depth": as_int(conf,"search_depth",32),
    }

    rec_str = conf.get("receptor","") or conf.get("receptor_file","")
    rec = Path(rec_str) if rec_str else (BASE / "receptors" / "target_prepared.pdbqt")
    if not rec.is_absolute():
        rec = (BASE/rec).resolve()
    if not rec.exists():
        raise SystemExit(f"❌ Receptor not found: {rec}")

    lig_dir = Path(conf["ligand_directory"]).resolve() if "ligand_directory" in conf else DIR_PREP
    out_dir = Path(conf["output_directory"]).resolve() if "output_directory" in conf else DIR_RESULTS

    chash = hashlib.sha1((cfg_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:10]

    print("Vina-GPU bin:", vgpu)
    print("Config:", cfg_path)
    print("Box:", box, "| GPU params:", gcfg)
    print("Ligand dir:", lig_dir, "| Output dir:", out_dir)
    return box,gcfg,rec,chash,lig_dir,out_dir,cfg_file

# -------------------------- Pose parsing --------------------------
RES_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.I)

def vina_pose_is_valid(p:Path)->Tuple[bool,Optional[float]]:
    try:
        if not p.exists() or p.stat().st_size<200:
            return (False,None)
        txt=p.read_text(errors="ignore")
        scores=[float(m.group(1)) for m in RES_RE.finditer(txt)]
        return ((len(scores)>0), (min(scores) if scores else None))
    except Exception:
        return (False,None)

# ------------------------------ Helpers ------------------------------
def chunked(it: Iterable[Path], n:int)->Iterable[list[Path]]:
    buf=[]
    for x in it:
        buf.append(x)
        if len(buf)==n:
            yield buf; buf=[]
    if buf:
        yield buf

def run_batch(vgpu:Path, cfg_file:Path, lig_dir:Path, out_dir:Path, gcfg:dict)->int:
    cmd=[str(vgpu),"--config",str(cfg_file),
         "--ligand_directory",str(lig_dir),
         "--output_directory",str(out_dir),
         "--thread",str(gcfg["thread"]),
         "--search_depth",str(gcfg["search_depth"])]
    print("Batch CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    env=os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX","2")
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        return 127

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vina-GPU worker — SHARD SAFE")
    ap.add_argument("--gpu", type=int, required=False, help="GPU ID to use (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--subset", type=str, default="", help="Path to file containing ligand IDs (one per line)")
    ap.add_argument("--out-manifest", type=str, default="", help="Per-GPU manifest shard path")
    ap.add_argument("--log", type=str, default="", help="Optional log file to tee stdout/stderr")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎯 Using GPU {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)

    if args.out_manifest:
        per_manifest = Path(args.out_manifest)
    else:
        tag = f"{args.gpu}" if args.gpu is not None else "X"
        per_manifest = DIR_STATE / f"manifest_gpu{tag}.csv"

    if args.log:
        sys.stdout = open(args.log, "a", buffering=1, encoding="utf-8")
        sys.stderr = sys.stdout

    # Locate binary and runtime config
    vgpu = None
    cfg_file = None
    try:
        # We reuse load_runtime from above; to keep this self-contained,
        # replicate minimal logic inline for cfg_file if load_runtime signature differs.
        vgpu = None  # we will set below to pass linter
    except Exception:
        pass

    # Reconstruct simplified load_runtime (avoid ref mismatch)
    # (We duplicate logic to ensure this worker is self-contained.)
    # Find binary
    candidates = [
        BASE / "AutoDock-Vina-GPU-2-1",
        shutil.which("AutoDock-Vina-GPU-2-1") and Path(shutil.which("AutoDock-Vina-GPU-2-1")),
        BASE / "vina-gpu",
        BASE / "Vina-GPU",
    ]
    for p in candidates:
        if not p: continue
        p = Path(p)
        if p.exists():
            try:
                mode = p.stat().st_mode
                if not (mode & 0o111):
                    p.chmod(mode | 0o755)
            except Exception:
                pass
            vgpu = p.resolve(); break
    if not vgpu:
        raise SystemExit("❌ AutoDock-Vina-GPU-2-1 binary not found in project root or PATH.")

    # Resolve config
    cfg_gpu = vgpu.parent/"VinaGPUConfig.txt"
    cfg_cpu = vgpu.parent/"VinaConfig.txt"
    cfg_path = cfg_gpu if cfg_gpu.exists() else cfg_cpu
    if not cfg_path.exists():
        raise SystemExit(f"❌ Config not found: {cfg_path}")
    conf={}
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        if "#" in line: line=line.split("#",1)[0].strip()
        if "=" not in line: continue
        k,v=line.split("=",1); conf[k.strip().lower()]=v.strip()

    def as_float(d,k,default): 
        try: return float(d.get(k,default))
        except: return float(default)
    def as_int(d,k,default):
        try: return int(str(d.get(k,default)).strip())
        except: return int(default)

    box = {
        "center_x": as_float(conf,"center_x",0.0),
        "center_y": as_float(conf,"center_y",0.0),
        "center_z": as_float(conf,"center_z",0.0),
        "size_x":   as_float(conf,"size_x",20.0),
        "size_y":   as_float(conf,"size_y",20.0),
        "size_z":   as_float(conf,"size_z",20.0),
    }
    gcfg = {
        "thread": max(1000, as_int(conf,"thread",10000)),
        "search_depth": as_int(conf,"search_depth",32),
    }
    rec_str = conf.get("receptor","") or conf.get("receptor_file","")
    rec = Path(rec_str) if rec_str else (BASE / "receptors" / "target_prepared.pdbqt")
    if not rec.is_absolute(): rec = (BASE/rec).resolve()
    if not rec.exists(): raise SystemExit(f"❌ Receptor not found: {rec}")
    lig_dir = Path(conf["ligand_directory"]).resolve() if "ligand_directory" in conf else DIR_PREP
    out_dir = Path(conf["output_directory"]).resolve() if "output_directory" in conf else DIR_RESULTS
    chash = hashlib.sha1((cfg_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:10]

    print("Vina-GPU bin:", vgpu)
    print("Config:", cfg_path)
    print("Box:", box, "| GPU params:", gcfg)
    print("Ligand dir:", lig_dir, "| Output dir:", out_dir)

    # Enumerate ligands
    all_ligs = sorted(lig_dir.glob("*.pdbqt"))
    if not all_ligs:
        raise SystemExit("❌ No ligand PDBQTs found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional subset list
    subset_ids = set()
    if args.subset:
        p = Path(args.subset)
        if not p.exists():
            raise SystemExit(f"❌ Subset file not found: {p}")
        subset_ids = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
        if subset_ids:
            all_ligs = [p for p in all_ligs if p.stem in subset_ids]
            print(f"📄 Subset active: {len(subset_ids)} IDs → {len(all_ligs)} ligand files", flush=True)

    # SAFE_RESUME
    if SAFE_RESUME:
        pending=[p for p in all_ligs if not (out_dir / f"{p.stem}_out.pdbqt").exists()]
    else:
        pending=list(all_ligs)

    # Load or init per-GPU shard manifest
    manifest = load_manifest(per_manifest) if per_manifest.exists() else {}

    created_ts = now_iso()
    receptor_sha = sha1_of_file(rec)

    # Filter invalid atom types
    valid_pending=[]
    for lig in pending:
        ok, why = pdbqt_has_only_allowed_types(lig)
        if not ok:
            lig_id = lig.stem
            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"]=lig_id
            m["pdbqt_path"]=str(lig.resolve())
            m["vina_status"]="FAILED"
            m["vina_reason"]=why
            m.setdefault("created_at", created_ts)
            m["updated_at"]=now_iso()
            manifest[lig_id]=m
            print(f"⚠️  Skipping {lig.name} — {why}", flush=True)
        else:
            valid_pending.append(lig)
    save_manifest(per_manifest, manifest)

    if not valid_pending:
        print("✅ No valid ligands left to process for this worker.", flush=True)
        print(f"Per-GPU manifest: {per_manifest}", flush=True)
        return

    # Mini-batch loop
    tag = f"gpu{args.gpu}" if args.gpu is not None else "gpuX"
    tmp_root = out_dir / "_batch_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        for bi, batch in enumerate(chunked(valid_pending, BATCH_SIZE), 1):
            if STOP_REQUESTED or HARD_STOP:
                print("🧾 Stop requested — exiting before next batch.", flush=True)
                break

            tmp_dir = tmp_root / f"{tag}_b{bi:04d}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            for lig in batch:
                shutil.copy2(lig, tmp_dir / lig.name)

            rc = run_batch(vgpu, cfg_path, tmp_dir, out_dir, gcfg)
            if rc != 0:
                print(f"⚠️  Batch {bi} rc={rc}. Harvesting outputs then stopping.", flush=True)

            # Harvest
            for lig in batch:
                lig_id = lig.stem
                pose = out_dir / f"{lig_id}_out.pdbqt"
                ok, best = vina_pose_is_valid(pose)
                m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
                m["id"]=lig_id
                m["pdbqt_path"]=str(lig.resolve())
                m["vina_status"]="DONE" if ok else "FAILED"
                m["vina_pose"]=str(pose.resolve())
                m["vina_reason"]="OK" if ok else "No VINA RESULT found"
                m["vina_score"]=f"{best:.2f}" if ok and best is not None else ""
                m["config_hash"]=chash
                m["receptor_sha1"]=receptor_sha
                m["tools_vina"]=str(vgpu)
                m.setdefault("created_at", created_ts)
                m["updated_at"]=now_iso()
                manifest[lig_id]=m

            save_manifest(per_manifest, manifest)

            if not KEEP_TMP:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if rc != 0:
                break

    finally:
        if not KEEP_TMP:
            for p in (out_dir / "_batch_tmp").glob(f"{tag}_b*"):
                shutil.rmtree(p, ignore_errors=True)
        save_manifest(per_manifest, manifest)
        print("✅ Worker docking done (or safely stopped).", flush=True)
        print(f"Per-GPU manifest: {per_manifest}", flush=True)

if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYTHONUNBUFFERED","1")
    main()
