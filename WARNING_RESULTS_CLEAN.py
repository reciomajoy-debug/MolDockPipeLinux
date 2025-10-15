#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WARNING_CLEANUP_PIPELINE_v3c.py
Fast, idempotent cleaner for your Molecular Docking workspace (Windows-friendly).

What's new in v3c
-----------------
• Optional deletion of docked poses in results/: use --clean-results-pdbqt
  (Removes results/*.pdbqt whose filename token matches a completed ID; shallow only.)
• Keeps all final CSVs intact: results/leaderboard.csv, results/summary.csv, state/manifest.csv.

Summary
-------
1) Determines "completed" IDs:
   - DONE: ID in results/leaderboard.csv OR manifest shows Vina completion (status/score)
   - FAILED: any of {admet_status, sdf_status, pdbqt_status, vina_status} ∈ {FAILED, ERROR}
   - ADMET-processed: admet_status ∈ {PASSED, FAILED, SKIPPED_ADMET} (counts as completed for input.csv cleanup)
2) Deletes per-ID intermediates (idempotent; preserves final CSVs).
3) Updates input/input.csv (backs up unless --no-backup).
4) Optional: also delete results/*.pdbqt per ID via --clean-results-pdbqt

Usage
-----
Dry run preview:
    python WARNING_CLEANUP_PIPELINE_v3c.py --workspace . --show-exists --max-ids 200

Apply (all):
    python WARNING_CLEANUP_PIPELINE_v3c.py --workspace . --apply

Also remove docked PDBQTs in results/:
    python WARNING_CLEANUP_PIPELINE_v3c.py --workspace . --apply --clean-results-pdbqt

Only FAILED:
    python WARNING_CLEANUP_PIPELINE_v3c.py --workspace . --apply --only-failed

Other options:
    --id-col COL        # custom ID column
    --status-col COL    # optional hint; auto-checks vina/admet/sdf/pdbqt/status anyway
    --include-results-logs   # also delete results/*_{vina,vinagpu}.log
    --keep-sdf          # keep 3D_Structures/<id>.sdf
    --keep-pdbqt        # keep prepared_ligands/<id>.pdbqt
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import signal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# --------------------------- Ctrl+C handling ---------------------------
STOP = False
def _sigint(sig, frame):
    global STOP
    if not STOP:
        STOP = True
        print("\n⏹️  Ctrl+C — will stop after current checkpoint...", flush=True)
signal.signal(signal.SIGINT, _sigint)

# --------------------------- CSV helpers ---------------------------
def read_csv_ids(path: Path, id_cols: List[str]) -> Set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cols = [c for c in id_cols if c in fields] or (fields[:1] if fields else [])
        for row in reader:
            for c in cols:
                v = (row.get(c) or "").strip()
                if v:
                    ids.add(v)
                    break
    return ids

def read_manifest_done_failed(path: Path, id_cols: List[str], status_col_hint: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns (done_ids, failed_ids, admet_done_ids)
    - DONE if Vina evidence: vina_status ∈ {DONE, COMPLETED, SUCCESS} OR non-empty vina_score
    - FAILED if any of {admet_status, sdf_status, pdbqt_status, vina_status, status} ∈ {FAILED, ERROR}
    - ADMET-DONE if admet_status ∈ {PASSED, FAILED, SKIPPED_ADMET}
    """
    done, failed, admet_done = set(), set(), set()
    if not path.exists():
        return done, failed, admet_done
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        id_col = next((c for c in id_cols if c in fields), (fields[0] if fields else None))
        if id_col is None:
            return done, failed, admet_done

        known_status_cols = ["admet_status","sdf_status","pdbqt_status","vina_status","status"]
        if status_col_hint and status_col_hint not in known_status_cols:
            known_status_cols.insert(0, status_col_hint)

        for row in reader:
            _id = (row.get(id_col) or "").strip()
            if not _id:
                continue

            admet_status = (row.get("admet_status") or "").strip().upper()
            vina_status  = (row.get("vina_status")  or "").strip().upper()
            vina_score   = (row.get("vina_score")   or "").strip()

            if vina_status in {"DONE", "COMPLETED", "SUCCESS"} or vina_score not in {"", "NA", "N/A"}:
                done.add(_id)

            if admet_status in {"PASSED", "FAILED", "SKIPPED_ADMET"}:
                admet_done.add(_id)

            for sc in known_status_cols:
                val = (row.get(sc) or "").strip().upper()
                if val in {"FAILED", "ERROR"}:
                    failed.add(_id)
                    break

    return done, failed, admet_done

def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    b = path.with_suffix(path.suffix + f".bak-{ts}")
    shutil.copy2(path, b)
    return b

# --------------------------- Directory indexing ---------------------------
def index_files(dirpath: Path, include_dirs: bool = False) -> List[Path]:
    out: List[Path] = []
    if dirpath.exists():
        for entry in dirpath.iterdir():
            try:
                if entry.is_file() or (include_dirs and entry.is_dir()):
                    out.append(entry.resolve())
            except Exception:
                continue
    return out

def head_token(name: str) -> str:
    for sep in ("_", "."):
        if sep in name:
            return name.split(sep, 1)[0]
    return name

def build_prefix_map(paths: List[Path]) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    for p in paths:
        tok = head_token(p.name)
        m.setdefault(tok, []).append(p)
    return m

# --------------------------- File ops ---------------------------
def safe_delete(paths: Iterable[Path], dry_run: bool) -> Tuple[int, int]:
    files, bytes_ = 0, 0
    for p in paths:
        try:
            if p.exists():
                if p.is_dir():
                    if dry_run:
                        print(f"[DRY] RMDIR {p}")
                    else:
                        shutil.rmtree(p, ignore_errors=True)
                        print(f"RMDIR {p}")
                elif p.is_file():
                    size = p.stat().st_size
                    if dry_run:
                        print(f"[DRY] DELETE {p}")
                    else:
                        p.unlink()
                        print(f"DELETED {p}")
                    files += 1
                    bytes_ += size
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")
    return files, bytes_

# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Idempotent cleaner: remove intermediate files for DONE/FAILED (incl. ADMET) IDs.")
    ap.add_argument("--workspace", type=str, default=".", help="Path to workspace root (default: .)")
    ap.add_argument("--apply", action="store_true", help="Actually delete files (default: dry run)")
    ap.add_argument("--no-backup", action="store_true", help="Do not backup input CSV before editing")
    ap.add_argument("--only-failed", action="store_true", help="Only clean FAILED IDs (do not clean DONE)")
    ap.add_argument("--id-col", type=str, default=None, help="Custom ID column name if not id/ID/ligand_id/LigandID")
    ap.add_argument("--status-col", type=str, default="status", help="(Optional) status column hint for manifest")
    ap.add_argument("--results-dir", type=str, default="results", help="Where leaderboard/summary live (default: results)")
    ap.add_argument("--input-csv", type=str, default=str(Path("input") / "input.csv"), help="Path to input CSV")
    ap.add_argument("--max-ids", type=int, default=None, help="Only process the first N IDs (for testing)")
    ap.add_argument("--show-exists", action="store_true", help="Print counts of existing files per category before deletion")
    ap.add_argument("--include-results-logs", action="store_true", help="Also delete results/*_{vina,vinagpu}.log per ID")
    ap.add_argument("--clean-results-pdbqt", action="store_true", help="Also delete results/*.pdbqt that match completed IDs")
    ap.add_argument("--keep-sdf", action="store_true", help="Keep 3D_Structures/<id>.sdf files (do not delete)")
    ap.add_argument("--keep-pdbqt", action="store_true", help="Keep prepared_ligands/<id>.pdbqt files (do not delete)")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    results_dir = ws / args.results_dir
    leaderboard_csv = results_dir / "leaderboard.csv"
    manifest_csv    = ws / "state" / "manifest.csv"
    input_csv       = (ws / args.input_csv) if not Path(args.input_csv).is_absolute() else Path(args.input_csv)

    id_cols = [args.id_col] if args.id_col else ["id", "ID", "Id", "ligand_id", "LigandID"]

    # Determine completed IDs
    done_ids_leader = read_csv_ids(leaderboard_csv, id_cols)
    m_done, m_failed, m_admet_done = read_manifest_done_failed(manifest_csv, id_cols, args.status_col)

    done_ids = done_ids_leader | m_done
    failed_ids = m_failed

    # "Completed" for deletion
    completed_for_delete = (failed_ids if args.only_failed else (done_ids | failed_ids))
    # "Completed" for input includes ADMET-processed too
    completed_for_input = completed_for_delete | m_admet_done

    print(f"Workspace : {ws}")
    print(f"results   : {results_dir}")
    print(f"input.csv : {input_csv}")
    print(f"DONE IDs  : {len(done_ids)}")
    print(f"FAILED IDs: {len(failed_ids)}")
    print(f"ADMET IDs : {len(m_admet_done)}  (treated as completed for input.csv)")
    print(f"TOTAL (delete set): {len(completed_for_delete)}")

    # Optional limit for testing
    ids_delete = sorted(completed_for_delete)
    ids_input  = sorted(completed_for_input)
    if args.max_ids is not None:
        ids_delete = ids_delete[:max(0, int(args.max_ids))]
        ids_input  = ids_input[:max(0, int(args.max_ids))]
        print(f"Limiting to first {len(ids_delete)} IDs due to --max-ids.")

    if not ids_delete and not ids_input:
        print("Nothing to clean. Exiting.")
        return

    # Index directories once (shallow)
    d_struct = ws / "3D_Structures"
    d_prep   = ws / "prepared_ligands"
    d_output = ws / "output"
    d_state  = ws / "state"
    d_ckpt   = ws / "_pipeline_checkpoints"
    d_results= ws / "results"

    struct_files = index_files(d_struct)
    prep_files   = index_files(d_prep)
    out_files    = index_files(d_output)
    state_files  = [p for p in index_files(d_state) if p.name not in {"manifest.csv","admet_pass.list","admet_fail.list"}]
    ckpt_entries = index_files(d_ckpt, include_dirs=True)
    results_files= index_files(d_results)

    # Build lookups
    map_struct = {p.name: p for p in struct_files}  # exact names
    map_prep   = {p.name: p for p in prep_files}
    map_out    = build_prefix_map(out_files)
    map_state  = build_prefix_map(state_files)
    map_ckpt   = build_prefix_map(ckpt_entries)

    # Optional: results logs
    map_reslogs = {}
    if args.include_results_logs:
        res_logs = [p for p in results_files if p.suffix.lower()==".log" and ("_vina" in p.stem or "_vinagpu" in p.stem)]
        map_reslogs = build_prefix_map(res_logs)

    # Optional: results PDBQT deletion
    map_respdbqt = {}
    if args.clean_results_pdbqt:
        res_pdbqts = [p for p in results_files if p.suffix.lower()==".pdbqt"]
        map_respdbqt = build_prefix_map(res_pdbqts)

    # Preview counts
    if args.show_exists:
        c_sdf = c_smi = c_rdlog = c_pdbqt = c_out = c_state = c_ckpt = c_rlogs = c_rpdbqt = 0
        for cid in ids_delete:
            if not args.keep_sdf and f"{cid}.sdf" in map_struct: c_sdf += 1
            if f"{cid}.smi" in map_struct: c_smi += 1
            if f"{cid}_rdkit.log" in map_struct: c_rdlog += 1
            if not args.keep_pdbqt and f"{cid}.pdbqt" in map_prep: c_pdbqt += 1
            c_out   += len(map_out.get(cid, []))
            c_state += len(map_state.get(cid, []))
            c_ckpt  += len(map_ckpt.get(cid, []))
            if args.include_results_logs:
                c_rlogs += len(map_reslogs.get(cid, []))
            if args.clean_results_pdbqt:
                c_rpdbqt += len(map_respdbqt.get(cid, []))
        print(f"Existing to delete -> SDF:{c_sdf} SMI:{c_smi} RDLOG:{c_rdlog} PDBQT:{c_pdbqt} OUT:{c_out} STATE:{c_state} CKPT:{c_ckpt} RESLOG:{c_rlogs} RES_PDBQT:{c_rpdbqt}")

    # Collect deletions in batches
    BATCH = 400
    batch: List[Path] = []
    deleted_files = 0
    freed_bytes = 0
    total_planned = 0

    def enqueue(p: Optional[Path]):
        nonlocal total_planned
        if p and p.exists():
            batch.append(p)
            total_planned += 1

    def flush():
        nonlocal deleted_files, freed_bytes, batch
        files, bytes_ = safe_delete(batch, dry_run=(not args.apply))
        deleted_files += files
        freed_bytes += bytes_
        batch = []

    # Build target list to DELETE (ids_delete)
    for idx, cid in enumerate(ids_delete, 1):
        if STOP:
            print("🧾 Stop requested — finalizing after current checkpoint...")
            break

        # 3D_Structures
        if not args.keep_sdf: enqueue(map_struct.get(f"{cid}.sdf"))
        enqueue(map_struct.get(f"{cid}.smi"))
        enqueue(map_struct.get(f"{cid}_rdkit.log"))

        # prepared_ligands
        if not args.keep_pdbqt: enqueue(map_prep.get(f"{cid}.pdbqt"))

        # output/state/ckpt (shallow)
        for p in map_out.get(cid, []):   enqueue(p)
        for p in map_state.get(cid, []): enqueue(p)
        for p in map_ckpt.get(cid, []):  enqueue(p)

        # results logs (optional)
        if args.include_results_logs:
            for p in map_reslogs.get(cid, []): enqueue(p)

        # results PDBQT (optional)
        if args.clean_results_pdbqt:
            for p in map_respdbqt.get(cid, []): enqueue(p)

        if idx % BATCH == 0:
            print(f"… processed {idx} IDs; deleting batch of {len(batch)} items")
            flush()

    # final batch
    if batch:
        print(f"Final batch: deleting {len(batch)} items")
        flush()

    if args.apply:
        print(f"Deleted files: {deleted_files} | Freed ~{freed_bytes/1_048_576:.2f} MiB")
    else:
        print(f"[DRY] Would delete ~{total_planned} item(s).")

    # Update input.csv — remove IDs that are completed for input (ids_input)
    if input_csv.exists():
        if not args.no_backup:
            b = backup_file(input_csv)
            if b:
                print(f"Backed up input CSV -> {b.name}")
        try:
            with input_csv.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
            id_col = next((c for c in id_cols if c in fieldnames), (fieldnames[0] if fieldnames else None))
            if id_col is None:
                print(f"[WARN] Could not determine ID column in {input_csv}; skipping input update.")
            else:
                ids_this_run = set(ids_input)
                kept = [r for r in rows if (r.get(id_col) or "").strip() not in ids_this_run]
                removed_n = len(rows) - len(kept)
                if not args.apply:
                    print(f"[DRY] Would remove {removed_n} row(s) from {input_csv}")
                else:
                    with input_csv.open("w", newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in kept:
                            writer.writerow(r)
                    print(f"Updated {input_csv} (removed {removed_n} row(s))")
        except Exception as e:
            print(f"[WARN] Could not update {input_csv}: {e}")
    else:
        print(f"[INFO] {input_csv} not found; skipping input update")

    print("✅ Cleanup finished (v3c). Final CSVs in results/ and state/manifest.csv are preserved.")
    print("Re-run is safe (idempotent). Use --apply to actually delete.")

if __name__ == "__main__":
    main()
