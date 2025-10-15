#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WARNING_PURGE_INCOMPLETE_VINA.py
Reset manifest entries that claim a finished/failed docking but have no valid pose.
- Conservatively resets:
    • vina_status == DONE but pose missing/invalid  → reset to PENDING
    • vina_status == FAILED and pose missing/invalid → reset to PENDING
- Rebuilds results/summary.csv and results/leaderboard.csv from manifest.
- Writes a TODO list of ligands to (re)dock: state/vina_todo.list
- Makes a timestamped backup of state/manifest.csv before changes.

Run:
    python WARNING_PURGE_INCOMPLETE_VINA.py
"""

from __future__ import annotations
import csv
import shutil
from pathlib import Path
from datetime import datetime, timezone
import re
from typing import Optional, Tuple

BASE = Path(".").resolve()
DIR_STATE   = BASE / "state"
DIR_RESULTS = BASE / "results"
DIR_PREP    = BASE / "prepared_ligands"

FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_SUMMARY  = DIR_RESULTS / "summary.csv"
FILE_LEADER   = DIR_RESULTS / "leaderboard.csv"
FILE_TODO     = DIR_STATE / "vina_todo.list"

# --- CSV fields (match Modules 1–4) ---
MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]

# --- Helpers ---
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv_dicts(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

def backup_manifest(src: Path) -> Optional[Path]:
    if not src.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = src.with_name(f"manifest.backup.{ts}.csv")
    shutil.copy2(src, dst)
    return dst

# Pose validation used in Modules 4a/4b
VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.I)
def vina_pose_is_valid(path: Path) -> Tuple[bool, Optional[float]]:
    try:
        if not path.exists() or path.stat().st_size < 200:
            return (False, None)
        txt = path.read_text(errors="ignore")
        scores = [float(m.group(1)) for m in VINA_RESULT_RE.finditer(txt)]
        if not scores:
            return (False, None)
        return (True, min(scores))
    except Exception:
        return (False, None)

def resolve_pose_path(mrow: dict) -> Path:
    """Prefer manifest's vina_pose; fall back to results/<id>_out.pdbqt."""
    pose_str = (mrow.get("vina_pose") or "").strip()
    if pose_str:
        p = Path(pose_str)
        if not p.is_absolute():
            p = (BASE / p).resolve()
        return p
    # fallback conventional name
    return (DIR_RESULTS / f"{mrow.get('id','').strip()}_out.pdbqt").resolve()

def build_summaries_from_manifest(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    # summary.csv
    summary_headers = ["id","inchikey","vina_score","pose_path","created_at"]
    summary_rows = []
    for m in sorted(rows, key=lambda r: r.get("id","")):
        sc = m.get("vina_score","")
        if sc:
            summary_rows.append({
                "id": m.get("id",""),
                "inchikey": m.get("inchikey",""),
                "vina_score": sc,
                "pose_path": m.get("vina_pose",""),
                "created_at": m.get("updated_at","") or m.get("created_at","")
            })
    # leaderboard.csv
    leader_headers = ["rank","id","inchikey","vina_score","pose_path"]
    ranked = sorted(summary_rows, key=lambda r: float(r["vina_score"])) if summary_rows else []
    leaders = []
    for i, r in enumerate(ranked, 1):
        leaders.append({
            "rank": i,
            "id": r["id"],
            "inchikey": r["inchikey"],
            "vina_score": r["vina_score"],
            "pose_path": r["pose_path"]
        })
    return (
        summary_rows,   # matches summary_headers
        leaders         # matches leader_headers
    )

def main():
    if not FILE_MANIFEST.exists():
        raise SystemExit("❌ state/manifest.csv not found.")

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    DIR_STATE.mkdir(parents=True, exist_ok=True)

    rows = read_csv_dicts(FILE_MANIFEST)
    if not rows:
        print("ℹ️ manifest is empty. Nothing to do.")
        write_csv_dicts(FILE_SUMMARY, [], ["id","inchikey","vina_score","pose_path","created_at"])
        write_csv_dicts(FILE_LEADER,  [], ["rank","id","inchikey","vina_score","pose_path"])
        FILE_TODO.write_text("", encoding="utf-8")
        return

    # Backup first
    bkup = backup_manifest(FILE_MANIFEST)
    if bkup:
        print(f"📦 Backup written: {bkup.name}")

    fixes = 0
    todo_ids = set()

    for m in rows:
        lig_id = (m.get("id") or "").strip()
        if not lig_id:
            continue

        # If there's a prepared ligand, it's eligible to dock.
        prepared_ok = (DIR_PREP / f"{lig_id}.pdbqt").exists()

        vstat = (m.get("vina_status") or "").strip().upper()
        pose  = resolve_pose_path(m)
        valid, best = vina_pose_is_valid(pose)

        # Rule 1: DONE but missing/invalid → reset
        if vstat == "DONE" and not valid:
            m["vina_status"] = ""
            m["vina_reason"] = "RESET: missing/invalid pose"
            m["vina_score"]  = ""
            m["vina_pose"]   = ""   # clear stale path
            m["updated_at"]  = now_iso()
            fixes += 1
            if prepared_ok:
                todo_ids.add(lig_id)

        # Rule 2: FAILED but missing/invalid → allow retry (reset)
        elif vstat == "FAILED" and not valid:
            m["vina_status"] = ""
            m["vina_reason"] = "RESET: previously failed, pose missing/invalid"
            m["vina_score"]  = ""
            m["vina_pose"]   = ""
            m["updated_at"]  = now_iso()
            fixes += 1
            if prepared_ok:
                todo_ids.add(lig_id)

        # Rule 3: DONE and valid → ensure score present
        elif vstat == "DONE" and valid:
            if not (m.get("vina_score") or "").strip() and best is not None:
                m["vina_score"] = f"{best:.2f}"
                m["updated_at"] = now_iso()

        # Rule 4: anything else without a valid result but with prepared ligand → queue
        elif prepared_ok and not valid:
            # Not marking as reset here if status already empty; just ensure queued.
            todo_ids.add(lig_id)

        # Normalize relative pose path into manifest if file is under ./results
        if valid and pose.is_file():
            try:
                rel = pose.relative_to(BASE)
                m["vina_pose"] = str(rel).replace("\\", "/")
            except Exception:
                # keep absolute if outside
                m["vina_pose"] = str(pose)

    # Save manifest (preserving headers/order)
    # Ensure all fields exist per row
    fixed_rows = []
    for m in rows:
        fixed = {k: m.get(k, "") for k in MANIFEST_FIELDS}
        fixed_rows.append(fixed)
    write_csv_dicts(FILE_MANIFEST, fixed_rows, MANIFEST_FIELDS)

    # Write TODO list for Module 4
    FILE_TODO.write_text("\n".join(sorted(todo_ids)) + ("\n" if todo_ids else ""), encoding="utf-8")

    # Rebuild summaries from manifest
    summary_rows, leader_rows = build_summaries_from_manifest(fixed_rows)
    write_csv_dicts(FILE_SUMMARY, summary_rows, ["id","inchikey","vina_score","pose_path","created_at"])
    write_csv_dicts(FILE_LEADER, leader_rows, ["rank","id","inchikey","vina_score","pose_path"])

    print(f"✅ Scan complete. Resets applied: {fixes}")
    print(f"   Docking TODO count: {len(todo_ids)} → {FILE_TODO}")
    print(f"   Summary rebuilt: {FILE_SUMMARY}")
    print(f"   Leaderboard rebuilt: {FILE_LEADER}")
    if fixes:
        print("   Tip: Re-run Module 4a/4b; only VALID 'DONE' entries will be skipped now.")

if __name__ == "__main__":
    main()
