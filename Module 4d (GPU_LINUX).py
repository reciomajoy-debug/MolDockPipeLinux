#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4b (GPU) — Linux mirror for AutoDock‑Vina‑GPU‑2‑1

Goal:
  Drop‑in replacement of your existing Module 4b (GPU) mini‑batch docker,
  but targeting the Linux binary: ./AutoDock‑Vina‑GPU‑2‑1

Behavior:
  • Idempotent resume: skips ligands that already have *_out.pdbqt
  • Mini‑batch launcher: copies only the current batch into a temp lig dir
  • Graceful stop: Ctrl+C once = finish current batch; twice = stop ASAP
  • Atom‑type sanity check (AD4 types) before docking
  • Manifest/summary/leaderboard kept in sync after every batch
  • Linux‑friendly: no Windows .exe assumptions; chmod +x if needed

Inputs (same as 4b):
  ./prepared_ligands/*.pdbqt
  ./VinaGPUConfig.txt  (preferred) or ./VinaConfig.txt (fallback)
  ./receptors/target_prepared.pdbqt  (fallback if not set in config)

Outputs:
  ./results/<lig>_out.pdbqt (poses)
  ./state/manifest.csv      (merged updates)
  ./results/summary.csv and ./results/leaderboard.csv

Run:
  python3 "Module 4b (GPU) — Linux (AutoDock‑Vina‑GPU‑2‑1).py"

Notes:
  • Expects the Vina‑GPU 2.1 Linux binary named exactly: AutoDock‑Vina‑GPU‑2‑1
  • Reads ligand_directory/output_directory from config if present; otherwise
    defaults to ./prepared_ligands and ./results respectively.
  • Keeps per‑batch temp under results/_batch_tmp/ (deleted by default).
"""

from __future__ import annotations
import csv, hashlib, os, re, shlex, shutil, signal, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Iterable

# ======================== Tunables ========================
BATCH_SIZE  = 64      # ligands per batch
SAFE_RESUME = True    # skip ligs that already have *_out.pdbqt
KEEP_TMP    = False   # keep per‑batch folder (debug)
# =========================================================

STOP_REQUESTED = False
HARD_STOP = False

def _sigint(_, __):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C — finishing current batch then exiting cleanly…")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will exit ASAP after harvest.")

signal.signal(signal.SIGINT, _sigint)

BASE = Path(".").resolve()
DIR_PREP      = BASE / "prepared_ligands"
DIR_RESULTS   = BASE / "results"
DIR_STATE     = BASE / "state"
DIR_REC_FALLBACK = BASE / "receptors" / "target_prepared.pdbqt"

FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_SUMMARY  = DIR_RESULTS / "summary.csv"
FILE_LEADER   = DIR_RESULTS / "leaderboard.csv"

for d in (DIR_RESULTS, DIR_STATE):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------ Utils ------------------------------

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


def load_manifest()->dict[str,dict]:
    if not FILE_MANIFEST.exists():
        return {}
    out={}
    for r in read_csv(FILE_MANIFEST):
        row={k:r.get(k,"") for k in MANIFEST_FIELDS}
        out[row.get("id"," ")] = row
    return out


def save_manifest(m:dict[str,dict])->None:
    rows=[{k:v.get(k,"") for k,v in [(key, row) for key,row in m.items()]} for _ in ()]
    # The above trick flattens; clearer version:
    rows = [{k:v.get(k,"") for k in MANIFEST_FIELDS} for _,v in sorted(m.items())]
    atomic_write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# -------------------------- Atom‑types check --------------------------
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
        BASE / "vina-gpu",  # permissive fallbacks
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
    raise SystemExit("❌ AutoDock‑Vina‑GPU‑2‑1 binary not found in project root or PATH.")


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
        # Vina‑GPU 2.1 prefers large thread count; keep sane minimum
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

    print("Vina‑GPU bin:", vgpu)
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


def build_and_write_summaries(manifest: dict[str,dict])->None:
    summ_headers=["id","inchikey","vina_score","pose_path","created_at"]
    rows=[]
    for _,m in sorted(manifest.items()):
        sc=m.get("vina_score","")
        if sc:
            rows.append({
                "id":m.get("id",""),
                "inchikey":m.get("inchikey",""),
                "vina_score":sc,
                "pose_path":m.get("vina_pose",""),
                "created_at":m.get("updated_at","")
            })
    atomic_write_csv(FILE_SUMMARY, rows, summ_headers)

    lead_headers=["rank","id","inchikey","vina_score","pose_path"]
    ranked=sorted(rows, key=lambda r: float(r["vina_score"])) if rows else []
    leaders=[{"rank":i,
              "id":r["id"],
              "inchikey":r["inchikey"],
              "vina_score":r["vina_score"],
              "pose_path":r["pose_path"]} for i,r in enumerate(ranked,1)]
    atomic_write_csv(FILE_LEADER, leaders, lead_headers)

# ------------------------------ Runner ------------------------------

def run_batch(vgpu:Path, cfg_file:Path, lig_dir:Path, out_dir:Path, gcfg:dict)->int:
    cmd=[str(vgpu),"--config",str(cfg_file),
         "--ligand_directory",str(lig_dir),
         "--output_directory",str(out_dir),
         "--thread",str(gcfg["thread"]),
         "--search_depth",str(gcfg["search_depth"])]

    print("Batch CMD:", " ".join(shlex.quote(c) for c in cmd))

    # Linux niceness: keep system responsive
    env=os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX","2")  # reduce glibc heap arenas
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        return 127

# ------------------------------ Main ------------------------------

def main():
    vgpu = find_vinagpu_binary()
    box,gcfg,receptor,chash,lig_dir,out_dir,cfg_file = load_runtime(vgpu)

    # enumerate ligands
    all_ligs = sorted(lig_dir.glob("*.pdbqt"))
    if not all_ligs:
        raise SystemExit("❌ No ligand PDBQTs found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAFE_RESUME: skip ligands that already have outputs
    if SAFE_RESUME:
        pending=[p for p in all_ligs if not (out_dir / f"{p.stem}_out.pdbqt").exists()]
    else:
        pending=list(all_ligs)

    manifest = load_manifest()
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
            print(f"⚠️  Skipping {lig.name} — {why}")
        else:
            valid_pending.append(lig)
    save_manifest(manifest)

    if not valid_pending:
        print("✅ No valid ligands left to process. Summaries updated.")
        build_and_write_summaries(manifest)
        return

    # Mini‑batch loop under results/_batch_tmp
    tmp_root = out_dir / "_batch_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        for bi, batch in enumerate(chunked(valid_pending, BATCH_SIZE), 1):
            if STOP_REQUESTED or HARD_STOP:
                print("🧾 Stop requested — exiting before next batch.")
                break

            tmp_dir = tmp_root / f"b{bi:04d}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Copy only small PDBQT files (fast on Linux too; avoids symlink issues)
            for lig in batch:
                shutil.copy2(lig, tmp_dir / lig.name)

            rc = run_batch(vgpu, cfg_file, tmp_dir, out_dir, gcfg)
            if rc != 0:
                print(f"⚠️  Batch {bi} rc={rc}. Harvesting outputs then stopping.")

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

            save_manifest(manifest)
            build_and_write_summaries(manifest)

            if not KEEP_TMP:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if rc != 0:
                break

    finally:
        if not KEEP_TMP:
            shutil.rmtree(tmp_root, ignore_errors=True)
        save_manifest(manifest)
        build_and_write_summaries(manifest)
        print("✅ Mini‑batch GPU docking done (or safely stopped).")
        print(f"Manifest: {FILE_MANIFEST}")
        print(f"Summary : {FILE_SUMMARY}")
        print(f"Leaders : {FILE_LEADER}")


if __name__ == "__main__":
    # Friendly preflight for Linux users
    if sys.platform.startswith("linux"):
        # Prefer python3 on PATH for subprocess children that might spawn
        os.environ.setdefault("PYTHONUNBUFFERED","1")
    main()
