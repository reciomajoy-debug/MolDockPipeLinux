#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4e Worker (GPU/Linux) — built from your current Linux GPU mini-batch runner
- Runs docking on ONE assigned GPU (via --gpu <id> or CUDA_VISIBLE_DEVICES)
- Keeps per-GPU isolation: temp dirs, manifest fragment, optional log
- Preserves: SAFE_RESUME, batching (64), atom-type checks, graceful stop
- Defers summary/leaderboard building to the controller

Usage (examples):
  python Module4e_worker.py --gpu 0
  python Module4e_worker.py --gpu 1 --subset state/subset_gpu1.list --out-manifest state/manifest_gpu1.csv --log logs/gpu1.log
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

# ======================== Tunables (same defaults) ========================
BATCH_SIZE  = 64      # ligands per batch
SAFE_RESUME = True    # skip ligs that already have *_out.pdbqt
KEEP_TMP    = False   # keep per-batch folder (debug)
# ========================================================================

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
DIR_PREP      = BASE / "prepared_ligands"
DIR_RESULTS   = BASE / "results"
DIR_STATE     = BASE / "state"
DIR_REC_FALLBACK = BASE / "receptors" / "target_prepared.pdbqt"

# Global (shared) manifest path is NOT written by workers.
# Controller will merge per-GPU manifests into this.
FILE_MANIFEST_GLOBAL = DIR_STATE / "manifest.csv"

for d in (DIR_RESULTS, DIR_STATE):
    d.mkdir(parents=True, exist_ok=True)

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
    if not path.exists():
        return {}
    out={}
    for r in read_csv(path):
        row={k:r.get(k,"") for k in MANIFEST_FIELDS}
        out[row.get("id"," ")] = row
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
    # Primary expected name for Linux build
    candidates = [
        BASE / "AutoDock-Vina-GPU-2-1",
        shutil.which("AutoDock-Vina-GPU-2-1") and Path(shutil.which("AutoDock-Vina-GPU-2-1")),
        BASE / "vina-gpu",     # permissive fallbacks
        BASE / "Vina-GPU",
    ]
    for p in candidates:
        if not p:
            continue
        p = Path(p)
        if p.exists():
            # Ensure executable on Linux
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
    rec = Path(rec_str) if rec_str else DIR_REC_FALLBACK
    if not rec.is_absolute():
        rec = (vgpu.parent/rec).resolve()
    if not rec.exists():
        raise SystemExit(f"❌ Receptor not found: {rec}")

    lig_dir = Path(conf["ligand_directory"]).resolve() if "ligand_directory" in conf else DIR_PREP
    out_dir = Path(conf["output_directory"]).resolve() if "output_directory" in conf else DIR_RESULTS

    chash = hashlib.sha1((cfg_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:10]

    print("Vina-GPU bin:", vgpu)
    print("Config:", cfg_path)
    print("Box:", box, "| GPU params:", gcfg)
    print("Ligand dir:", lig_dir, "| Output dir:", out_dir)

    return box,gcfg,rec,chash,lig_dir,out_dir,cfg_path

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
            yield buf
            buf=[]
    if buf:
        yield buf

# ------------------------------ Runner ------------------------------

def run_batch(vgpu:Path, cfg_file:Path, lig_dir:Path, out_dir:Path, gcfg:dict)->int:
    cmd=[str(vgpu),"--config",str(cfg_file),
         "--ligand_directory",str(lig_dir),
         "--output_directory",str(out_dir),
         "--thread",str(gcfg["thread"]),
         "--search_depth",str(gcfg["search_depth"])]

    print("Batch CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)

    # Linux niceness: keep system responsive
    env=os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX","2")  # reduce glibc heap arenas
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        return 127

# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Vina-GPU worker (Linux, single-GPU)")
    ap.add_argument("--gpu", type=int, required=False, help="GPU ID to use (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--subset", type=str, default="", help="Optional path to a file containing ligand IDs (one per line) to process in this worker")
    ap.add_argument("--out-manifest", type=str, default="", help="Optional override for per-GPU manifest path (default: state/manifest_gpu<gpu>.csv)")
    ap.add_argument("--log", type=str, default="", help="Optional log file path; if set, worker writes its stdout there as well")
    args = ap.parse_args()

    # Set CUDA_VISIBLE_DEVICES if --gpu is provided
    gpu_id = args.gpu
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🎯 Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)

    # Determine per-GPU manifest path
    if args.out_manifest:
        per_manifest_path = Path(args.out_manifest)
    else:
        # If GPU is None, still make a worker manifest
        tag = f"{gpu_id}" if gpu_id is not None else "X"
        per_manifest_path = DIR_STATE / f"manifest_gpu{tag}.csv"

    # Optional log tee (basic)
    if args.log:
        # naive tee: reopen stdout to append to log file
        sys.stdout = open(args.log, "a", buffering=1, encoding="utf-8")
        sys.stderr = sys.stdout

    vgpu = find_vinagpu_binary()
    box,gcfg,receptor,chash,lig_dir,out_dir,cfg_file = load_runtime(vgpu)

    # enumerate ligands
    all_ligs = sorted(lig_dir.glob("*.pdbqt"))
    if not all_ligs:
        raise SystemExit("❌ No ligand PDBQTs found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional subset filter (IDs one per line)
    subset_ids = set()
    if args.subset:
        p = Path(args.subset)
        if not p.exists():
            raise SystemExit(f"❌ Subset file not found: {p}")
        subset_ids = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
        if subset_ids:
            all_ligs = [p for p in all_ligs if p.stem in subset_ids]
            print(f"📄 Subset active: {len(subset_ids)} IDs → {len(all_ligs)} ligand files", flush=True)

    # SAFE_RESUME: skip ligands that already have outputs
    if SAFE_RESUME:
        pending=[p for p in all_ligs if not (out_dir / f"{p.stem}_out.pdbqt").exists()]
    else:
        pending=list(all_ligs)

    # Load OR initialize the per-GPU manifest snapshot (not the global)
    manifest = load_manifest(per_manifest_path) if per_manifest_path.exists() else {}

    created_ts = now_iso()
    receptor_sha = sha1_of_file(receptor)

    # Filter out ligands with invalid atom types (fail fast)
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
    save_manifest(per_manifest_path, manifest)

    if not valid_pending:
        print("✅ No valid ligands left to process for this worker.", flush=True)
        print(f"Per-GPU manifest: {per_manifest_path}", flush=True)
        return

    # Mini-batch loop under results/_batch_tmp/gpu<id>_b####
    tag = f"gpu{gpu_id}" if gpu_id is not None else "gpuX"
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

            # Copy only small PDBQT files (fast on Linux; avoids symlink issues)
            for lig in batch:
                shutil.copy2(lig, tmp_dir / lig.name)

            rc = run_batch(vgpu, cfg_file, tmp_dir, out_dir, gcfg)
            if rc != 0:
                print(f"⚠️  Batch {bi} rc={rc}. Harvesting outputs then stopping.", flush=True)

            # Harvest outputs for this batch
            for lig in batch:
                lig_id = lig.stem
                pose = out_dir / f"{lig_id}_out.pdbqt"
                ok,best = vina_pose_is_valid(pose)
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

            # Write ONLY the per-GPU manifest (controller merges later)
            save_manifest(per_manifest_path, manifest)

            if not KEEP_TMP:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if rc != 0:
                break

    finally:
        if not KEEP_TMP:
            # clean only our own GPU-tagged temp dirs; leave others intact
            for p in (out_dir / "_batch_tmp").glob(f"{tag}_b*"):
                shutil.rmtree(p, ignore_errors=True)

        # Final flush
        save_manifest(per_manifest_path, manifest)
        print("✅ Worker docking done (or safely stopped).", flush=True)
        print(f"Per-GPU manifest: {per_manifest_path}", flush=True)

if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYTHONUNBUFFERED","1")
    main()
