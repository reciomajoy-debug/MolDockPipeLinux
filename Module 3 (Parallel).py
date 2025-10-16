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
    meeko_cmd: mk_prepare_ligand.py        # or absolute path; we try smart fallbacks
    python_exe: python3.11                 # used for -m meeko.cli_prepare_ligand fallback

  policy:
    skip_if_done: true                     # skip if PDBQT exists & validates
    purge_old_meeko_logs: true             # delete old *.log in prepared_ligands
    quiet_subprocess: true                 # squelch meeko stdout/stderr

  parallel:
    enabled: true
    max_workers: null                      # auto: cores-1, capped at 8
    backend: process                       # process|thread (Linux default overridden to thread)
    checkpoint_every: 50                   # write manifest every N finished jobs
"""

from __future__ import annotations
import csv
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ------------------------------ Paths ----------------------------------------

ROOT = Path(__file__).resolve().parent
DIR_SDF = (ROOT / "3D_Structures").resolve()
DIR_PDBQT = (ROOT / "prepared_ligands").resolve()
DIR_CONFIG = (ROOT / "config").resolve()
DIR_STATE = (ROOT / "state").resolve()
FILE_MANIFEST = (DIR_STATE / "manifest.csv").resolve()
FILE_RUNYML = (DIR_CONFIG / "run.yml").resolve()
FILE_MACHINEYML = (DIR_CONFIG / "machine.yml").resolve()

DIR_PDBQT.mkdir(parents=True, exist_ok=True)
DIR_STATE.mkdir(parents=True, exist_ok=True)

# ------------------------------ Signals --------------------------------------

STOP_REQUESTED = False
HARD_STOP = False

def _handle_sigint(signum, frame):
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
    MEEKO_VER = getattr(meeko, "__version__", "")
except Exception:
    MEEKO_VER = ""

# ------------------------------ Utilities ------------------------------------

def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")

def deep_update(base: dict, other: dict) -> dict:
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: v for k, v in row.items()})
    return rows

def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

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
            "meeko_cmd": "mk_prepare_ligand.py",
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
            chunks.append(sha1_file(p))
    return hashlib.sha1(("|".join(chunks)).encode()).hexdigest()[:12]

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
    if not DIR_SDF.exists():
        return id2sdf
    for p in sorted(DIR_SDF.glob("*.sdf")):
        lig_id = p.stem
        id2sdf[lig_id] = p
        if lig_id not in manifest:
            manifest[lig_id] = {k: "" for k in MANIFEST_FIELDS}
            manifest[lig_id]["id"] = lig_id
            manifest[lig_id]["created_at"] = now_iso()
    return id2sdf

def pdbqt_is_valid(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        # simple fast check: first line should contain "REMARK" or "ROOT" or "ATOM"
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2000)
        if ("ROOT" in head) or ("ATOM" in head) or ("REMARK" in head):
            return True
        return False
    except Exception:
        return False

# ------------------------------ Meeko runner ---------------------------------

def run_meeko_quiet(meeko_cmd: str, python_exe: str, in_sdf: Path, out_pdbqt: Path, quiet: bool=True) -> Tuple[bool, str]:
    """
    Try multiple ways of invoking Meeko to produce PDBQT.
    Writes to out_pdbqt.tmp first, validates, then renames.
    Returns (ok, reason)
    """
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
        """
        Interruptible child process runner:
        - Starts Meeko in its own process group (start_new_session=True)
        - Polls so we can react to HARD_STOP quickly
        """
        import time
        import errno
        import signal as _signal

        try:
            proc = subprocess.Popen(
                cmd_list,
                stdout=(subprocess.DEVNULL if quiet else None),
                stderr=(subprocess.DEVNULL if quiet else None),
                shell=False,
                start_new_session=True,
            )
        except FileNotFoundError:
            return errno.ENOENT

        # polling loop
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            if HARD_STOP:
                try:
                    os.killpg(proc.pid, _signal.SIGTERM)
                except Exception:
                    pass
                t0 = time.time()
                while proc.poll() is None and time.time() - t0 < 2.0:
                    time.sleep(0.05)
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, _signal.SIGKILL)
                    except Exception:
                        pass
                return 128 + 15
            time.sleep(0.05)

    candidates = [
        [meeko_cmd, "-i", str(in_sdf), "-o", str(tmp_pdbqt)],
        ["mk_prepare_ligand", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],
        [python_exe, "-m", "meeko.cli_prepare_ligand", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],
    ]

    for cmd in candidates:
        try:
            rc = _exec(cmd)
            if rc == 0 and pdbqt_is_valid(tmp_pdbqt):
                tmp_pdbqt.replace(out_pdbqt)
                return True, f"OK via {cmd[0]}"
        except FileNotFoundError:
            continue
        except Exception:
            continue

    # Cleanup tmp on failure
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
    default_backend = "thread" if os.name == "posix" else "process"
    backend = str(par.get("backend", default_backend)).lower()
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
        raise SystemExit("❌ No SDF files found in 3D_Structures")

    # Purge old logs if requested
    if purge_old_logs:
        for p in DIR_PDBQT.glob("*.log"):
            try:
                p.unlink()
            except Exception:
                pass

    # Build job list
    jobs: list[tuple[str, str]] = []
    for lig_id, sdf_path in sorted(id2sdf.items()):
        mrow = manifest.get(lig_id) or {}
        out_pdbqt = (DIR_PDBQT / f"{lig_id}.pdbqt")
        if skip_if_done and out_pdbqt.exists() and pdbqt_is_valid(out_pdbqt):
            # mark as done
            mrow["pdbqt_status"] = "DONE"
            mrow["pdbqt_path"] = str(out_pdbqt.resolve())
            mrow["pdbqt_reason"] = "OK (pre-existing)"
            mrow["tools_meeko"] = MEEKO_VER
            mrow["updated_at"] = now_iso()
            manifest[lig_id] = mrow
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
            fut = executor.submit(worker_prepare, lig_id, sdf_path_str, meeko_cmd, python_exe,
                                  quiet_subprocess)
            futures.append(fut)
            submitted += 1
        return end

    try:
        with Executor(max_workers=max_workers) as executor:
            next_idx = 0
            # Prime the queue
            next_idx = submit_next_batch(next_idx, max_workers)

            while futures:
                if HARD_STOP:
                    # Hard stop: don’t submit more; fall through to drain/cancel
                    pass
                elif not STOP_REQUESTED and next_idx < len(jobs):
                    # Keep queue topped up
                    next_idx = submit_next_batch(next_idx, max_workers - sum(1 for f in futures if not f.done()))

                # Wait for something to finish
                for fut in as_completed(list(futures), timeout=0.25):
                    futures.remove(fut)
                    try:
                        lig_id, ok, reason, out_pdbqt, sdf_path = fut.result()
                    except Exception as e:
                        ok = False
                        lig_id = "?"
                        reason = f"worker exception: {e}"
                        out_pdbqt = ""
                        sdf_path = ""

                    # Update manifest row
                    if lig_id not in manifest:
                        manifest[lig_id] = {k: "" for k in MANIFEST_FIELDS}
                        manifest[lig_id]["id"] = lig_id
                        manifest[lig_id]["created_at"] = created_ts

                    row = manifest[lig_id]
                    row["pdbqt_path"] = out_pdbqt
                    row["sdf_path"] = sdf_path
                    row["tools_meeko"] = MEEKO_VER
                    row["config_hash"] = chash
                    row["updated_at"] = now_iso()

                    if ok:
                        row["pdbqt_status"] = "DONE"
                        row["pdbqt_reason"] = reason
                        done += 1
                    else:
                        row["pdbqt_status"] = "FAILED"
                        row["pdbqt_reason"] = reason
                        failed += 1

                    # Checkpoint
                    total = done + failed
                    if total % checkpoint_every == 0:
                        save_manifest(manifest)
                        print(f"… checkpoint: {total}/{len(jobs)} (done={done} failed={failed})")

                # User asked to stop → don’t submit more; drain queue
                if STOP_REQUESTED or HARD_STOP:
                    if not futures:  # drained
                        break

        # If stop requested, cancel anything not yet started
        try:
            if (STOP_REQUESTED or HARD_STOP) and futures:
                for _f in list(futures):
                    _f.cancel()
        except Exception:
            pass

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
