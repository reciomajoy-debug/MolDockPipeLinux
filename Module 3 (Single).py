#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3: SDF -> ligand PDBQT via Meeko (QUIET + GRACEFUL STOP)
- Inputs:
    - 3D_Structures/<id>.sdf
    - state/manifest.csv (optional)
- Outputs:
    - prepared_ligands/<id>.pdbqt
    - manifest updated (pdbqt_* fields, tools_meeko)

Behavior changes:
- No per-ligand Meeko logs are written (suppresses *_meeko.log).
- Subprocess stdout/stderr are discarded to avoid clutter.
- Ctrl+C once: finish current ligand, flush manifest, exit cleanly.
- Ctrl+C twice: exit loop ASAP after a safe checkpoint and flush manifest.
- On start, optionally purges old *_meeko.log in prepared_ligands.
"""

from __future__ import annotations



def run_meeko_prepare(infile, outfile, extra_args=None, quiet=False):
    """Cross-platform Meeko ligand preparation.
    Tries mk_prepare_ligand, then module forms (main_prepare_ligand / cli_prepare_ligand).
    Uses the same Python interpreter running this script.
    """
    if extra_args is None:
        extra_args = []
    # Normalize to strings
    infile = str(infile)
    outfile = str(outfile)

    # Try direct CLI first
    try:
        meeko_exe = shutil.which("mk_prepare_ligand")
    except Exception:
        meeko_exe = None

    pyexe = sys.executable or "python3"

    candidates = []
    if meeko_exe:
        candidates.append([meeko_exe, "-i", infile, "-o", outfile] + extra_args)

    # modern then legacy module paths
    candidates.append([pyexe, "-m", "meeko.main_prepare_ligand", "-i", infile, "-o", outfile] + extra_args)
    candidates.append([pyexe, "-m", "meeko.cli_prepare_ligand",  "-i", infile, "-o", outfile] + extra_args)

    last_err = None
    for cmd in candidates:
        try:
            # Smoke test with -h to see if command exists
            try:
                test_cmd = cmd[:]
                # replace input/output with -h for the smoke test
                test_cmd = test_cmd[:2] if test_cmd[0].endswith("mk_prepare_ligand") else test_cmd[:3]
                test_cmd += ["-h"]
                subprocess.run(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception:
                # If smoke test fails, try running anyway (some envs return nonzero for -h)
                pass

            if not quiet:
                print("▶ Preparing ligand via:", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(outfile):
                if not quiet:
                    print(f"✓ Ligand prepared: {outfile}")
                return True
            else:
                last_err = res.stderr or res.stdout
                if not quiet:
                    print(f"⚠️ Meeko returned {res.returncode} for {infile}\n{last_err}")
        except FileNotFoundError as e:
            last_err = str(e)
            continue
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError("""❌ Meeko CLI not found or failed to run.
Tried: mk_prepare_ligand, python -m meeko.main_prepare_ligand, python -m meeko.cli_prepare_ligand
Last error:
{}
Install/upgrade with:
    python3 -m pip install --upgrade meeko
Or install latest dev:
    python3 -m pip install --no-cache-dir git+https://github.com/forlilab/meeko.git
""".format(last_err if last_err else "(no stderr)"))

import shutil
import sys
import csv
import hashlib
import json
import shlex
import subprocess
import signal
from pathlib import Path
from datetime import datetime, timezone

# ------------------------------ Graceful stop --------------------------------
STOP_REQUESTED = False
HARD_STOP = False

def _handle_sigint(sig, frame):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C detected — finishing current ligand, then exiting cleanly...")
        print("   (Press Ctrl+C again to stop ASAP after checkpoint.)")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will abort the loop ASAP and finalize outputs.")

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
    # Defaults keep it quiet and fast
    cfg = {
        "tools": {
            "meeko_cmd": "mk_prepare_ligand.py.exe",  # or mk_prepare_ligand if on PATH
            "python_exe": "python"
        },
        "policy": {
            "skip_if_done": True,
            "purge_old_meeko_logs": True,   # delete lingering *_meeko.log on start
            "write_logs": False,            # do NOT write per-ligand logs
            "quiet_subprocess": True        # discard stdout/stderr from meeko
        }
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
    Try in order:
      1) meeko_cmd (default: mk_prepare_ligand.py.exe)
      2) mk_prepare_ligand
      3) python -m meeko.cli_prepare_ligand
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
        if quiet:
            res = subprocess.run(
                cmd_list,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False
            )
        else:
            res = subprocess.run(cmd_list, shell=False)
        return res.returncode

    # Candidate commands
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

    # Failure cleanup
    try:
        if tmp_pdbqt.exists():
            tmp_pdbqt.unlink()
    except Exception:
        pass
    return False, "All Meeko attempts failed"

# ------------------------------ Main -----------------------------------------
def main():
    cfg = load_config()
    chash = config_hash()
    meeko_cmd = str(cfg.get("tools", {}).get("meeko_cmd", "mk_prepare_ligand.py.exe"))
    python_exe = str(cfg.get("tools", {}).get("python_exe", "python"))
    skip_if_done = bool(cfg.get("policy", {}).get("skip_if_done", True))
    purge_old_logs = bool(cfg.get("policy", {}).get("purge_old_meeko_logs", True))
    quiet_subprocess = bool(cfg.get("policy", {}).get("quiet_subprocess", True))
    # write_logs is intentionally unused (always False here); we keep the knob for future.

    manifest = load_manifest()
    id2sdf = discover_sdf(manifest)

    if not id2sdf:
        raise SystemExit("❌ No SDFs found. Run Module 2 first.")

    # Optional: purge old *_meeko.log files that could interfere with Vina-GPU tools
    if purge_old_logs:
        for logf in DIR_PDBQT.glob("*_meeko.log"):
            try:
                logf.unlink()
            except Exception:
                pass

    done, failed = 0, 0
    created_ts = now_iso()

    try:
        for idx, (lig_id, sdf_path) in enumerate(sorted(id2sdf.items()), 1):
            if STOP_REQUESTED or HARD_STOP:
                print("🧾 Stop requested — finalizing after this checkpoint...")
                break

            out_pdbqt = (DIR_PDBQT / f"{lig_id}.pdbqt").resolve()

            # Skip only if an existing PDBQT validates
            if skip_if_done and pdbqt_is_valid(out_pdbqt):
                m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
                m["id"] = lig_id
                m["sdf_path"] = str(sdf_path)
                m["pdbqt_status"] = "DONE"
                m["pdbqt_path"] = str(out_pdbqt)
                m["pdbqt_reason"] = "Found existing valid PDBQT"
                m["config_hash"] = chash
                m["tools_meeko"] = MEEKO_VER or "Meeko"
                m.setdefault("created_at", created_ts)
                m["updated_at"] = now_iso()
                manifest[lig_id] = m
                done += 1
                if idx % 50 == 0:
                    save_manifest(manifest)
                continue

            ok, reason = run_meeko_quiet(meeko_cmd, python_exe, sdf_path, out_pdbqt, quiet=quiet_subprocess)

            # Update manifest
            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["sdf_path"] = str(sdf_path)
            m["pdbqt_status"] = "DONE" if ok else "FAILED"
            m["pdbqt_path"] = str(out_pdbqt)
            m["pdbqt_reason"] = "OK" if ok else reason
            m["config_hash"] = chash
            m["tools_meeko"] = MEEKO_VER or "Meeko"
            m.setdefault("created_at", created_ts)
            m["updated_at"] = now_iso()
            manifest[lig_id] = m

            if ok:
                done += 1
            else:
                failed += 1

            if idx % 50 == 0:
                save_manifest(manifest)

    finally:
        save_manifest(manifest)
        print(f"✅ SDF → PDBQT complete (or stopped). DONE: {done}  FAILED: {failed}")
        print(f"   Outputs in: {DIR_PDBQT}")
        print(f"   Manifest updated: {FILE_MANIFEST}")
        if STOP_REQUESTED or HARD_STOP:
            print("   (Exited early by user request.)")

if __name__ == "__main__":
    main()
