#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4e (Controller) — Multi-GPU orchestrator for Vina-GPU workers (Linux)

New in this version:
- Pre-flight consolidation BEFORE docking:
  * Merge any leftover per-GPU manifests (state/manifest_gpu*.csv) into state/manifest.csv
  * Backfill manifest rows from results/*_out.pdbqt (parses best score)
- Then proceed to split pending ligands, launch workers, and merge again at the end.

Usage examples:
  python "Module 4e (Controller).py"
  python "Module 4e (Controller).py" --gpu-ids 0,1,2,3
  python "Module 4e (Controller).py" --max-gpus 2
  python "Module 4e (Controller).py" --worker "Module 4e (Worker).py" --dry-run
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# ------------------------------ Paths ------------------------------
BASE = Path(".").resolve()
DIR_PREP    = BASE / "prepared_ligands"
DIR_RESULTS = BASE / "results"
DIR_STATE   = BASE / "state"
DIR_LOGS    = BASE / "logs"

FILE_MANIFEST      = DIR_STATE / "manifest.csv"
FILE_SUMMARY       = DIR_RESULTS / "summary.csv"
FILE_LEADERBOARD   = DIR_RESULTS / "leaderboard.csv"

for d in (DIR_PREP, DIR_RESULTS, DIR_STATE, DIR_LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------ CSV helpers ------------------------------
MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv_dicts(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def atomic_write_csv(path: Path, rows: List[dict], headers: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def load_manifest(path: Path) -> Dict[str, dict]:
    rows = read_csv_dicts(path)
    out: Dict[str, dict] = {}
    for r in rows:
        row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
        if row.get("id"):
            out[row["id"]] = row
    return out

def save_manifest(path: Path, manifest: Dict[str, dict]) -> None:
    rows = [{k: v.get(k, "") for k in MANIFEST_FIELDS} for _, v in sorted(manifest.items())]
    atomic_write_csv(path, rows, MANIFEST_FIELDS)

# ------------------------------ GPU detection ------------------------------
def detect_gpu_ids(max_gpus: int | None) -> List[int]:
    # Try nvidia-smi; fallback to 0..(max_gpus-1) or [0]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        ids = [int(x.strip()) for x in proc.stdout.splitlines() if x.strip().isdigit()]
        if max_gpus is not None:
            ids = ids[:max_gpus]
        if ids:
            return ids
    except Exception:
        pass
    if max_gpus is None:
        return [0]
    return list(range(max_gpus)) if max_gpus > 0 else [0]

def parse_gpu_ids_arg(arg: str) -> List[int]:
    ids = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok:
            ids.append(int(tok))
    return ids

# ------------------------------ Ligand splitting ------------------------------
def list_pending_ligands() -> List[Path]:
    """Pending = prepared_ligands/*.pdbqt with no results/<id>_out.pdbqt yet."""
    ligs = sorted(DIR_PREP.glob("*.pdbqt"))
    pending = []
    for p in ligs:
        out_pose = DIR_RESULTS / f"{p.stem}_out.pdbqt"
        if not out_pose.exists():
            pending.append(p)
    return pending

def round_robin_split(items: List[Path], buckets: List[int]) -> Dict[int, List[Path]]:
    m: Dict[int, List[Path]] = {b: [] for b in buckets}
    if not items:
        return m
    i = 0
    for it in items:
        m[buckets[i]].append(it)
        i = (i + 1) % len(buckets)
    return m

# ------------------------------ Summary/Leaderboard ------------------------------
def rebuild_summaries(global_manifest: Dict[str, dict]) -> None:
    # summary.csv: id, inchikey, vina_score, pose_path, created_at
    rows = []
    for _, m in sorted(global_manifest.items()):
        sc = m.get("vina_score", "")
        if sc:
            rows.append({
                "id": m.get("id", ""),
                "inchikey": m.get("inchikey", ""),
                "vina_score": sc,
                "pose_path": m.get("vina_pose", ""),
                "created_at": m.get("updated_at", ""),
            })
    atomic_write_csv(FILE_SUMMARY, rows, ["id","inchikey","vina_score","pose_path","created_at"])

    # leaderboard.csv sorted by ascending score
    def _safe_float(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return float("inf")

    ranked = sorted(rows, key=lambda r: _safe_float(r["vina_score"])) if rows else []
    leaders = []
    for idx, r in enumerate(ranked, 1):
        leaders.append({
            "rank": idx,
            "id": r["id"],
            "inchikey": r["inchikey"],
            "vina_score": r["vina_score"],
            "pose_path": r["pose_path"],
        })
    atomic_write_csv(FILE_LEADERBOARD, leaders, ["rank","id","inchikey","vina_score","pose_path"])

# ------------------------------ Subprocess management ------------------------------
def spawn_worker(worker_path: Path, gpu_id: int, subset_file: Path) -> subprocess.Popen:
    # Worker will also set CUDA_VISIBLE_DEVICES, but we set it here too
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    per_manifest = DIR_STATE / f"manifest_gpu{gpu_id}.csv"
    log_file = DIR_LOGS / f"gpu{gpu_id}.log"

    cmd = [
        sys.executable,
        str(worker_path),
        "--gpu", str(gpu_id),
        "--subset", str(subset_file),
        "--out-manifest", str(per_manifest),
        "--log", str(log_file),
    ]

    # Don't inherit stdout to avoid interleaving; worker writes its own log
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def forward_sigint_to(children: List[subprocess.Popen], grace_sec: float = 8.0) -> None:
    # Send SIGINT to each child; after grace, kill survivors
    for p in children:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    # Wait up to grace seconds
    t0 = time.time()
    while time.time() - t0 < grace_sec:
        if all((p.poll() is not None) for p in children):
            return
        time.sleep(0.25)
    # Force kill remaining
    for p in children:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

# ------------------------------ Pre-flight consolidation ------------------------------
RES_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.I)

def _best_score_from_pose(pose_path: Path) -> str:
    try:
        txt = pose_path.read_text(errors="ignore")
        scores = [float(m.group(1)) for m in RES_RE.finditer(txt)]
        return f"{min(scores):.2f}" if scores else ""
    except Exception:
        return ""

def _ts(s: str) -> str:
    return s or ""

def _merge_into(dest: Dict[str, dict], src_rows: List[dict]) -> Tuple[int,int]:
    """Merge rows into dest by id, keeping the row with latest updated_at.
       Returns (added, updated) counts."""
    added, updated = 0, 0
    for r in src_rows:
        rid = r.get("id", "")
        if not rid:
            continue
        row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
        if rid in dest:
            old = dest[rid]
            if _ts(row.get("updated_at","")) >= _ts(old.get("updated_at","")):
                dest[rid] = row
                updated += 1
        else:
            dest[rid] = row
            added += 1
    return added, updated

def backfill_from_results(manifest: Dict[str, dict]) -> Tuple[int,int]:
    """Ensure every results/*_out.pdbqt has a manifest row. Fill missing vina_* fields."""
    added, updated = 0, 0
    for pose in DIR_RESULTS.glob("*_out.pdbqt"):
        lig_id = pose.name[:-len("_out.pdbqt")]
        # prep path (if exists)
        pdbqt = DIR_PREP / f"{lig_id}.pdbqt"

        best = _best_score_from_pose(pose)
        if lig_id in manifest:
            row = manifest[lig_id]
            # Fill if empty / improve if missing
            changed = False
            if not row.get("vina_pose"):
                row["vina_pose"] = str(pose.resolve())
                changed = True
            if best and not row.get("vina_score"):
                row["vina_score"] = best
                changed = True
            if not row.get("vina_status"):
                row["vina_status"] = "DONE"
                changed = True
            if not row.get("pdbqt_path") and pdbqt.exists():
                row["pdbqt_path"] = str(pdbqt.resolve())
                changed = True
            if changed:
                row["updated_at"] = now_iso()
                updated += 1
        else:
            row = {k: "" for k in MANIFEST_FIELDS}
            row["id"] = lig_id
            if pdbqt.exists():
                row["pdbqt_path"] = str(pdbqt.resolve())
            row["vina_status"] = "DONE"
            row["vina_pose"] = str(pose.resolve())
            row["vina_score"] = best
            ts = now_iso()
            row["created_at"] = ts
            row["updated_at"] = ts
            manifest[lig_id] = row
            added += 1
    return added, updated

def preflight_consolidate() -> None:
    """Before launching workers: merge leftover per-GPU manifests into global,
       then backfill from results to cover the 'outputs exist but no manifest row' edge case."""
    print("🧹 Pre-flight: consolidating any leftover manifests and results…")
    # Start from existing global manifest (if any)
    merged = load_manifest(FILE_MANIFEST) if FILE_MANIFEST.exists() else {}

    # Merge any state/manifest_gpu*.csv
    per_gpu_files = sorted(DIR_STATE.glob("manifest_gpu*.csv"))
    total_added, total_updated = 0, 0
    for part in per_gpu_files:
        rows = read_csv_dicts(part)
        a,u = _merge_into(merged, rows)
        total_added += a
        total_updated += u

    # Backfill from results/_out.pdbqt
    b_add, b_upd = backfill_from_results(merged)

    # Save and build summaries
    save_manifest(FILE_MANIFEST, merged)
    rebuild_summaries(merged)

    print(f"   Merged per-GPU manifests: +{total_added} added, {total_updated} updated")
    print(f"   Backfilled from results : +{b_add} added, {b_upd} updated")
    print(f"   Global manifest          : {FILE_MANIFEST}")
    print(f"   Summary / Leaderboard    : {FILE_SUMMARY} / {FILE_LEADERBOARD}")

# ------------------------------ Merge manifests (end-of-run) ------------------------------
def merge_per_gpu_manifests_all(base_manifest: Dict[str, dict]) -> Dict[str, dict]:
    """End-of-run merge: glob all manifest_gpu*.csv regardless of GPU IDs."""
    merged = dict(base_manifest)
    per_gpu_files = sorted(DIR_STATE.glob("manifest_gpu*.csv"))
    for part in per_gpu_files:
        rows = read_csv_dicts(part)
        _merge_into(merged, rows)
    return merged

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Module 4e (Controller) — multi-GPU orchestrator")
    ap.add_argument("--worker", type=str, default="Module 4e (Worker).py",
                    help="Path to the worker script to run per GPU")
    ap.add_argument("--gpu-ids", type=str, default="",
                    help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). If omitted, auto-detect.")
    ap.add_argument("--max-gpus", type=int, default=4,
                    help="When auto-detecting, cap to at most this many GPUs (default: 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only; don’t launch workers")
    args = ap.parse_args()

    worker_path = Path(args.worker)
    if not worker_path.exists():
        raise SystemExit(f"❌ Worker script not found: {worker_path}")

    # 0) PRE-FLIGHT CONSOLIDATION (new)
    preflight_consolidate()

    # 1) GPUs to use
    if args.gpu_ids.strip():
        gpu_ids = parse_gpu_ids_arg(args.gpu_ids.strip())
    else:
        gpu_ids = detect_gpu_ids(args.max_gpus)
    if not gpu_ids:
        raise SystemExit("❌ No GPUs detected or specified.")
    print(f"🎯 GPUs to use: {gpu_ids}")

    # 2) Pending ligands AFTER consolidation/backfill
    pending = list_pending_ligands()
    total_ligs = len(sorted(DIR_PREP.glob("*.pdbqt")))
    print(f"📦 Ligands: total={total_ligs}  pending={len(pending)}")

    splits = round_robin_split(pending, gpu_ids)

    # 3) Write subset files
    subset_files: Dict[int, Path] = {}
    for gid in gpu_ids:
        subset = splits.get(gid, [])
        subset_path = DIR_STATE / f"subset_gpu{gid}.list"
        with subset_path.open("w", encoding="utf-8") as f:
            for p in subset:
                f.write(p.stem + "\n")
        subset_files[gid] = subset_path
        print(f"  • GPU {gid}: {len(subset)} ligands  →  {subset_path.name}")

    if args.dry_run:
        print("📝 Dry run — not launching workers.")
        return

    # 4) Spawn workers (skip empty subsets)
    procs: List[subprocess.Popen] = []
    try:
        for gid in gpu_ids:
            subset_path = subset_files[gid]
            p = spawn_worker(worker_path, gid, subset_path)
            procs.append(p)

        # 5) Wait for all
        print("🚀 Workers launched. Waiting for completion…  (Ctrl+C to stop gracefully)")
        while True:
            done = [p.poll() is not None for p in procs]
            if all(done):
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n⏹️  Ctrl+C — requesting workers to stop…")
        forward_sigint_to(procs)
    finally:
        # 6) END-OF-RUN MERGE + BACKFILL (again, to be extra safe)
        base = load_manifest(FILE_MANIFEST) if FILE_MANIFEST.exists() else {}
        merged = merge_per_gpu_manifests_all(base)
        b_add, b_upd = backfill_from_results(merged)
        save_manifest(FILE_MANIFEST, merged)
        rebuild_summaries(merged)

        # Report statuses
        rcs = [p.poll() for p in procs]
        print("✅ Controller done. Worker return codes:", rcs)
        print(f"   Manifest : {FILE_MANIFEST}  (backfill +{b_add}/{b_upd})")
        print(f"   Summary  : {FILE_SUMMARY}")
        print(f"   Leaders  : {FILE_LEADERBOARD}")

if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYTHONUNBUFFERED","1")
    main()
