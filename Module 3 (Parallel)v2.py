#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3 — Ligand Preparation (Cross-Platform, macOS-friendly)
- Reads 3D_Structures/*.sdf (or .mol/.mol2) produced by Module 2
- Converts to prepared_ligands/*.pdbqt using Meeko
- Updates state/manifest.csv with pdbqt_status/path/reason and tool info
- Robust Meeko detection: mk_prepare_ligand, mk_prepare_ligand.py,
  python -m meeko.main_prepare_ligand, python -m meeko.cli_prepare_ligand
"""

import csv
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except Exception as e:
    print("This module requires pandas. Install with: python3 -m pip install pandas")
    raise

BASE = Path(".").resolve()
DIR_3D = BASE / "3D_Structures"
DIR_OUT = BASE / "prepared_ligands"
STATE = BASE / "state"
MANIFEST = STATE / "manifest.csv"

DIR_OUT.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)

# ------------------ Meeko cross-platform caller ------------------
def _resolve_meeko_cmd(infile, outfile, extra_args=None):
    """
    Return a list of candidate command lists to run Meeko on any OS.
    Tries (in order):
      1) mk_prepare_ligand
      2) mk_prepare_ligand.py
      3) python -m meeko.main_prepare_ligand
      4) python -m meeko.cli_prepare_ligand
    """
    if extra_args is None:
        extra_args = []
    infile = str(infile)
    outfile = str(outfile)
    pyexe = sys.executable or "python3"

    candidates = []

    exe = shutil.which("mk_prepare_ligand")
    if exe:
        candidates.append([exe, "-i", infile, "-o", outfile, *extra_args])

    exe_py = shutil.which("mk_prepare_ligand.py")
    if exe_py:
        candidates.append([exe_py, "-i", infile, "-o", outfile, *extra_args])

    candidates.append([pyexe, "-m", "meeko.main_prepare_ligand", "-i", infile, "-o", outfile, *extra_args])
    candidates.append([pyexe, "-m", "meeko.cli_prepare_ligand", "-i", infile, "-o", outfile, *extra_args])

    return candidates


def run_meeko_prepare(infile, outfile, extra_args=None, quiet=False):
    """Try each candidate invocation until one succeeds and outfile exists."""
    last_err = None
    for cmd in _resolve_meeko_cmd(infile, outfile, extra_args):
        try:
            if not quiet:
                print("▶ Preparing ligand via:", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(outfile):
                if not quiet:
                    print(f"✓ Ligand prepared: {outfile}")
                return True, ""
            else:
                last_err = (res.stderr or res.stdout or "").strip()
                if not quiet:
                    print(f"⚠️ Meeko returned {res.returncode}\n{last_err}")
        except FileNotFoundError as e:
            last_err = str(e)
            continue
        except Exception as e:
            last_err = str(e)
            continue

    return False, (last_err or "(no stderr)")


# ------------------ Manifest helpers ------------------
def load_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            pass
    # minimal manifest if absent/corrupt
    cols = [
        "id", "smiles", "inchikey",
        "admet_status", "admet_reason",
        "sdf_status", "sdf_path", "sdf_reason",
        "pdbqt_status", "pdbqt_path", "pdbqt_reason",
        "tools_meeko", "updated_at"
    ]
    return pd.DataFrame(columns=cols)


def save_manifest(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def find_row(df: pd.DataFrame, ligand_id: str):
    if "id" in df.columns:
        m = df["id"].astype(str) == str(ligand_id)
        idx = df.index[m]
        if len(idx) > 0:
            return idx[0]
    return None


# ------------------ Main pipeline ------------------
def main():
    sdf_files = sorted(list(DIR_3D.glob("*.sdf"))) + \
                sorted(list(DIR_3D.glob("*.mol"))) + \
                sorted(list(DIR_3D.glob("*.mol2")))

    if not sdf_files:
        print("❌ No input SDF/MOL found in 3D_Structures. Run Module 2 first.")
        return 1

    df = load_manifest(MANIFEST)

    done = 0
    failed = 0
    failures_log_rows = []

    for sdf in sdf_files:
        lig_id = sdf.stem
        out_pdbqt = DIR_OUT / f"{lig_id}.pdbqt"

        ok, err = run_meeko_prepare(sdf, out_pdbqt, extra_args=None, quiet=False)

        row_idx = find_row(df, lig_id)
        if row_idx is None:
            # append new row skeleton
            row = {
                "id": lig_id,
                "sdf_path": str(sdf.relative_to(BASE)) if sdf.exists() else "",
                "sdf_status": "DONE",
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            row_idx = df.index[-1]

        df.loc[row_idx, "updated_at"] = datetime.now().isoformat(timespec="seconds")

        if ok:
            df.loc[row_idx, "pdbqt_status"] = "DONE"
            df.loc[row_idx, "pdbqt_path"] = str(out_pdbqt.relative_to(BASE))
            df.loc[row_idx, "pdbqt_reason"] = ""
            df.loc[row_idx, "tools_meeko"] = "auto"
            done += 1
        else:
            df.loc[row_idx, "pdbqt_status"] = "FAILED"
            df.loc[row_idx, "pdbqt_path"] = ""
            short_err = (err.splitlines()[0] if err else "").strip()
            df.loc[row_idx, "pdbqt_reason"] = short_err[:240]
            failures_log_rows.append([lig_id, str(sdf), err])
            failed += 1

    save_manifest(df, MANIFEST)

    if failures_log_rows:
        fail_csv = STATE / "module3_failures.csv"
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ligand_id", "sdf_path", "stderr"])
            w.writerows(failures_log_rows)
        print(f"⚠️ Failure details saved to {fail_csv}")

    print(f"\nSDF ➜ PDBQT complete (or stopped). DONE: {done}  FAILED: {failed}")
    print(f"  Outputs in: {DIR_OUT}")
    print(f"  Manifest updated: {MANIFEST}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
