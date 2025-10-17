#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3 (Parallel): SDF -> ligand PDBQT via Meeko
- Parallel workers using concurrent.futures (default: ProcessPool)
- Preserves Module 3 behavior (quiet subprocess, atomic writes, graceful stop)
- Inputs:
    - 3D_Structures/<id>.sdf
    - state/manifest.csv (optional)
- Outputs:
    - prepared_ligands/<id>.pdbqt
    - manifest updated (pdbqt_* fields, tools_meeko)

Config knobs (config/run.yml or config/machine.yml):

  tools:
    meeko_cmd: mk_prepare_ligand.py.exe   # or mk_prepare_ligand
    python_exe: python
  policy:
    skip_if_done: true
    purge_old_meeko_logs: true
    quiet_subprocess: true
  parallel:
    enabled: true
    max_workers: 4         # default: min(8, max(1, cpu_count()-1))
    backend: process       # process | thread
    checkpoint_every: 50   # save manifest every N completions

Run:  python "Module 3 (Parallel).py"
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

# ------------------------------ Graceful stop --------------------------------
STOP_REQUESTED = False
HARD_STOP = False

def _handle_sigint(sig, frame):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C detected — will stop submitting new jobs, finish in-flight, then exit cleanly…")
        print("   (Press Ctrl+C again to stop ASAP after current checkpoint.)")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will attempt to cancel pending work and finalize outputs.")

signal.signal(signal.SIGINT, _handle_sigint)

# ------------------------------ Optional: Meeko ver --------------------------
try:
    import meeko  # type: ignore
    MEEKO_VER = getattr(meeko, "__version__", "unknown")
except Exception:
    MEEKO_VER = ""

# ------------------------------ Optional YAML --------------------------------
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# ------------------------------ Paths ----------------------------------------
BASE = Path(".").resolve()
DIR_INPUT = BASE / "input"
DIR_STATE = BASE / "state"
DIR_SDF = BASE / "3D_Structures"
DIR_PDBQT = BASE / "prepared_ligands"
DIR_LOGS = BASE / "logs"

FILE_INPUT = DIR_INPUT / "input.csv"
FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_RUNYML = BASE / "config" / "run.yml"
FILE_MACHINEYML = BASE / "config" / "machine.yml"

for d in (DIR_PDBQT, DIR_LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------ Helpers --------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})


def deep_update(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v


def load_yaml(path: Path) -> dict:
    if not (yaml and path.exists()):
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_config() -> dict:
    cfg = {
        "tools": {
            "meeko_cmd": "mk_prepare_ligand.py.exe",
            "python_exe": "python",
        },
        "policy": {
            "skip_if_done": True,
            "purge_old_meeko_logs": True,
            "quiet_subprocess": True,
        },
        "parallel": {
            "enabled": True,
            "max_workers": None,  # will be resolved later
            "backend": "process",  # or "thread"
            "checkpoint_every": 50,
        },
    }
    deep_update(cfg, load_yaml(FILE_RUNYML))
    deep_update(cfg, load_yaml(FILE_MACHINEYML))
    return cfg


def config_hash() -> str:
    chunks = []
    for p in (FILE_RUNYML, FILE_MACHINEYML):
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8"))
    if not chunks:
        chunks.append("{}")
    return hashlib.sha1("||".join(chunks).encode("utf-8")).hexdigest()[:10]

# ------------------------------ Manifest -------------------------------------
MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]


def load_manifest() -> dict[str, dict]:
    if not FILE_MANIFEST.exists():
        return {}
    rows = read_csv(FILE_MANIFEST)
    out = {}
    for r in rows:
        row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
        out[row["id"]] = row
    return out


def save_manifest(manifest: dict[str, dict]) -> None:
    rows = [{k: v.get(k, "") for k in MANIFEST_FIELDS} for _, v in sorted(manifest.items())]
    write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# ------------------------------ Discovery ------------------------------------

def discover_sdf(manifest: dict[str, dict]) -> dict[str, Path]:
    id2sdf: dict[str, Path] = {}
    for lig_id, row in manifest.items():
        p = (row.get("sdf_path") or "").strip()
        if p:
            path = Path(p)
            if not path.is_absolute():
                path = (BASE / p).resolve()
            if path.exists():
                id2sdf[lig_id] = path
    for sdf in sorted(DIR_SDF.glob("*.sdf")):
        lig_id = sdf.stem
        id2sdf.setdefault(lig_id, sdf.resolve())
    return id2sdf

# ------------------------------ Validation -----------------------------------

def pdbqt_is_valid(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 200:
            return False
        txt = path.read_text(errors="ignore")
        has_atom = ("ATOM " in txt) or ("HETATM" in txt)
        has_tors = "TORSDOF" in txt
        return has_atom and has_tors
    except Exception:
        return False

# ------------------------------ Meeko call (QUIET) ---------------------------

def run_meeko_quiet(meeko_cmd: str, python_exe: str, in_sdf: Path, out_pdbqt: Path,
                    quiet: bool = True) -> tuple[bool, str]:
    """
    Cross-platform Meeko caller with widened compatibility:
      1) meeko_cmd (from config; e.g., mk_prepare_ligand.py.exe or mk_prepare_ligand)
      2) mk_prepare_ligand
      3) mk_prepare_ligand.py            # <— added for Linux/mac installs into ~/.local/bin
      4) python -m meeko.main_prepare_ligand
      5) python -m meeko.cli_prepare_ligand

    Writes to out_pdbqt.tmp first, validates, then renames.
    Returns (ok, reason)
    """
    import shutil

    in_sdf = in_sdf.resolve()
    out_pdbqt = out_pdbqt.resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdbqt = out_pdbqt.with_suffix(".pdbqt.tmp")

    # Clean stale outputs
    for p in (out_pdbqt, tmp_pdbqt):
        try:
            if Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass

    def _exec(cmd_list: list[str]) -> int:
        if quiet:
            res = subprocess.run(
                cmd_list,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
        else:
            res = subprocess.run(cmd_list, shell=False)
        return res.returncode

    # Prefer actual paths on PATH if available
    def _maybe(cmd):
        exe = shutil.which(cmd[0]) if len(cmd) and not os.path.sep in cmd[0] else cmd[0]
        if exe:
            cmd = [exe] + cmd[1:]
        return cmd

    py = python_exe or "python"

    candidates = [
        [meeko_cmd, "-i", str(in_sdf), "-o", str(tmp_pdbqt)],                # config override
        ["mk_prepare_ligand", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],      # typical console entry
        ["mk_prepare_ligand.py", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],   # Linux/mac in ~/.local/bin
        [py, "-m", "meeko.main_prepare_ligand", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],  # newer module path
        [py, "-m", "meeko.cli_prepare_ligand",  "-i", str(in_sdf), "-o", str(tmp_pdbqt)],  # legacy module path
    ]

    for raw in candidates:
        cmd = _maybe(raw)
        try:
            rc = _exec(cmd)
            if rc == 0 and pdbqt_is_valid(tmp_pdbqt):
                tmp_pdbqt.replace(out_pdbqt)
                return True, f"OK via {cmd[0]}"
        except FileNotFoundError:
            continue
        except Exception:
            continue

    try:
        if tmp_pdbqt.exists():
            tmp_pdbqt.unlink()
    except Exception:
        pass
    return False, "All Meeko attempts failed"
# ------------------------------ Worker ---------------------------------------

def worker_prepare(lig_id: str, sdf_path_str: str, meeko_cmd: str, python_exe: str, quiet: bool) -> Tuple[str, bool, str, str, str]:
    """Isolated worker for parallel execution.
    Returns: (lig_id, ok, reason, out_pdbqt_path, sdf_path)
    """
    sdf_path = Path(sdf_path_str)
    out_pdbqt = (DIR_PDBQT / f"{lig_id}.pdbqt").resolve()
    ok, reason = run_meeko_quiet(meeko_cmd, python_exe, sdf_path, out_pdbqt, quiet=quiet)
    return lig_id, ok, reason, str(out_pdbqt), str(sdf_path)

# ------------------------------ Main -----------------------------------------

def main():
    cfg = load_config()
    chash = config_hash()
    meeko_cmd = str(cfg.get("tools", {}).get("meeko_cmd", "mk_prepare_ligand.py.exe"))
    python_exe = str(cfg.get("tools", {}).get("python_exe", "python"))

    policy = cfg.get("policy", {})
    skip_if_done = bool(policy.get("skip_if_done", True))
    purge_old_logs = bool(policy.get("purge_old_meeko_logs", True))
    quiet_subprocess = bool(policy.get("quiet_subprocess", True))

    par = cfg.get("parallel", {})
    par_enabled = bool(par.get("enabled", True))
    backend = str(par.get("backend", "process")).lower()
    max_workers = par.get("max_workers")
    if not max_workers:
        # reasonable default, keep one core free; cap at 8 unless user raises it
        try:
            cpu = os.cpu_count() or 2
        except Exception:
            cpu = 2
        max_workers = max(1, min(8, cpu - 1))
    checkpoint_every = int(par.get("checkpoint_every", 50))

    manifest = load_manifest()
    id2sdf = discover_sdf(manifest)
    if not id2sdf:
        raise SystemExit("❌ No SDFs found. Run Module 2 first.")

    # Optional: purge old *_meeko.log that can confuse downstream tools
    if purge_old_logs:
        for logf in DIR_PDBQT.glob("*_meeko.log"):
            try:
                logf.unlink()
            except Exception:
                pass

    # Build job list, honoring skip_if_done
    jobs: list[tuple[str, str]] = []  # (lig_id, sdf_path_str)
    for lig_id, sdf_path in sorted(id2sdf.items()):
        out_pdbqt = (DIR_PDBQT / f"{lig_id}.pdbqt").resolve()
        if skip_if_done and pdbqt_is_valid(out_pdbqt):
            m = manifest.get(lig_id, {k: "" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["sdf_path"] = str(sdf_path)
            m["pdbqt_status"] = "DONE"
            m["pdbqt_path"] = str(out_pdbqt)
            m["pdbqt_reason"] = "Found existing valid PDBQT"
            m["config_hash"] = chash
            m["tools_meeko"] = MEEKO_VER or "Meeko"
            m.setdefault("created_at", now_iso())
            m["updated_at"] = now_iso()
            manifest[lig_id] = m
            continue
        jobs.append((lig_id, str(sdf_path)))

    if not jobs:
        save_manifest(manifest)
        print("✅ Nothing to do (all PDBQTs valid). Manifest refreshed.")
        return

    print(f"🧵 Parallel Meeko enabled: backend={backend} workers={max_workers} jobs={len(jobs)}")

    done = failed = 0
    created_ts = now_iso()

    # Choose executor
    Executor = ThreadPoolExecutor if backend.startswith("thread") else ProcessPoolExecutor

    # Submit in waves so we can honor Ctrl+C cleanly
    futures = []
    submitted = 0

    def submit_next_batch(start_idx: int, batch_size: int) -> int:
        nonlocal futures, submitted
        end = min(start_idx + batch_size, len(jobs))
        for i in range(start_idx, end):
            lig_id, sdf_path_str = jobs[i]
            fut = executor.submit(worker_prepare, lig_id, sdf_path_str, meeko_cmd, python_exe, quiet_subprocess)
            futures.append(fut)
            submitted += 1
        return end

    batch_size = max_workers  # keep queue roughly one wave ahead

    try:
        with Executor(max_workers=max_workers) as executor:
            next_idx = submit_next_batch(0, batch_size)

            # Consume as completed; keep the pipeline full unless stop requested
            while futures:
                for fut in as_completed(list(futures)):
                    futures.remove(fut)
                    try:
                        lig_id, ok, reason, out_pdbqt_path, sdf_path = fut.result()
                    except Exception as e:
                        # A worker crashed before returning
                        lig_id = "?"
                        ok = False
                        reason = f"Worker error: {e}"
                        out_pdbqt_path = ""
                        sdf_path = ""

                    # Update manifest row atomically in main process
                    if lig_id != "?":
                        m = manifest.get(lig_id, {k: "" for k in MANIFEST_FIELDS})
                        m["id"] = lig_id
                        if sdf_path:
                            m["sdf_path"] = sdf_path
                        m["pdbqt_status"] = "DONE" if ok else "FAILED"
                        m["pdbqt_path"] = out_pdbqt_path
                        m["pdbqt_reason"] = "OK" if ok else reason
                        m["config_hash"] = chash
                        m["tools_meeko"] = MEEKO_VER or "Meeko"
                        m.setdefault("created_at", created_ts)
                        m["updated_at"] = now_iso()
                        manifest[lig_id] = m
                        done += int(ok)
                        failed += int(not ok)

                    # Periodic checkpoints
                    total = done + failed
                    if total and (total % checkpoint_every == 0):
                        save_manifest(manifest)
                        print(f"📒 Checkpoint — DONE: {done}  FAILED: {failed}")

                    # Top up queue if not stopping
                    if not STOP_REQUESTED and not HARD_STOP and next_idx < len(jobs):
                        next_idx = submit_next_batch(next_idx, batch_size)

                # If stop requested, let current futures drain without adding new ones
                if STOP_REQUESTED or HARD_STOP:
                    if not futures:  # drained
                        break

    finally:
        # Final flush
        save_manifest(manifest)
        print(f"✅ Parallel Meeko complete (or stopped). DONE: {done}  FAILED: {failed}")
        print(f"   Outputs in: {DIR_PDBQT}")
        print(f"   Manifest updated: {FILE_MANIFEST}")
        if STOP_REQUESTED or HARD_STOP:
            print("   (Exited early by user request.)")


if __name__ == "__main__":
    main()
