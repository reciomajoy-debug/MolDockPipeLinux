#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Update Helper for Molecular Docking System
-------------------------------------------------
This script ensures that the following CSVs exist and are properly formatted:
    • state/manifest.csv
    • results/summary.csv
    • results/leaderboard.csv

- If missing, they are created with the correct headers.
- If existing, they are updated in-place (headers preserved, data kept).
- Compatible with Module 1–4 schema.
"""

import csv
from pathlib import Path

BASE = Path(".").resolve()
STATE = BASE / "state"
RESULTS = BASE / "results"

FILE_MANIFEST = STATE / "manifest.csv"
FILE_SUMMARY = RESULTS / "summary.csv"
FILE_LEADER = RESULTS / "leaderboard.csv"

# ---- Canonical Headers ----
HEADERS_MANIFEST = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]

HEADERS_SUMMARY = ["id","inchikey","vina_score","pose_path","created_at"]
HEADERS_LEADER = ["rank","id","inchikey","vina_score","pose_path"]

# ---- Helpers ----
def ensure_csv(path: Path, headers: list[str], truncate: bool = False):
    """Ensure CSV exists and has headers. If truncate=True, clears all rows."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or truncate:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
        print(f"📄 Created new CSV: {path.name}")
        return

    # validate and fix headers if needed
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
        print(f"🧹 Empty CSV fixed: {path.name}")
        return
    if rows[0] != headers:
        print(f"⚠️  Header mismatch in {path.name} — rewriting header only.")
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in rows[1:]:
                w.writerow(r)

def main():
    print("🔧 Forcing update of summary.csv, leaderboard.csv, and manifest.csv ...\n")
    ensure_csv(FILE_MANIFEST, HEADERS_MANIFEST, truncate=False)
    ensure_csv(FILE_SUMMARY, HEADERS_SUMMARY, truncate=False)
    ensure_csv(FILE_LEADER, HEADERS_LEADER, truncate=False)
    print("\n✅ Force update complete. All CSVs now have correct headers and structure.")

if __name__ == "__main__":
    main()
