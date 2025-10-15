#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean pipeline outputs for a fresh deployment.
- Preserves folder structure
- Ensures all CSVs exist with correct headers
- Truncates CSVs to headers only (rows cleared)
- Removes generated ligand/receptor/results files
- Does not touch VinaConfig.txt or Vina binaries
- Adds double confirmation before executing
"""

from pathlib import Path
import sys

BASE = Path(".").resolve()

# folders to clean
FOLDERS_TO_CLEAN = [
    "input",
    "output",
    "3D_Structures",
    "prepared_ligands",
    "results",
    "state"
]

# extensions to delete inside folders
DELETE_EXTS = {".smi", ".sdf", ".pdbqt", ".log", ".tmp"}

# protected files
KEEP_FILES = {"VinaConfig.txt"}

# required CSVs and their headers
CSV_HEADERS = {
    "state/manifest.csv": [
        "id","smiles","inchikey",
        "admet_status","admet_reason",
        "sdf_status","sdf_path","sdf_reason",
        "pdbqt_status","pdbqt_path","pdbqt_reason",
        "vina_status","vina_score","vina_pose","vina_reason",
        "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
        "created_at","updated_at"
    ],
    "results/summary.csv": [
        "id","inchikey","vina_score","pose_path","created_at"
    ],
    "results/leaderboard.csv": [
        "rank","id","inchikey","vina_score","pose_path"
    ]
}


def truncate_or_create_csv(file: Path, headers: list[str]):
    """Truncate CSV to headers only, or create if missing."""
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(",".join(headers) + "\n", encoding="utf-8")
    action = "Truncated" if file.exists() else "Created new"
    print(f"📄 {action} CSV: {file}")


def clean_folder(folder: Path):
    """Recursively delete unwanted files."""
    if not folder.exists() or not folder.is_dir():
        return
    for f in folder.glob("*"):
        if f.is_file():
            if f.suffix.lower() == ".csv":
                continue  # handled separately
            if f.name in KEEP_FILES:
                continue
            if f.suffix.lower() in DELETE_EXTS:
                print(f"🗑️ Deleting {f}")
                try:
                    f.unlink()
                except Exception as e:
                    print(f"  ⚠️ Could not delete {f}: {e}")
        elif f.is_dir():
            clean_folder(f)


def confirm_action():
    """Ask for double confirmation before proceeding."""
    print(f"\n🔍 Base Directory: {BASE}")
    print("This operation will:")
    print(" - Clean folders:", ", ".join(FOLDERS_TO_CLEAN))
    print(" - Delete .smi, .sdf, .pdbqt, .log, .tmp files")
    print(" - Truncate or recreate manifest and result CSVs")
    print(" - Preserve VinaConfig.txt and binaries\n")

    confirm1 = input(f"Are you sure you want to purge '{BASE}'? (y/N): ").strip().lower()
    if confirm1 != "y":
        print("❌ Operation cancelled at first confirmation.")
        sys.exit(0)

    confirm2 = input("Really sure? This will delete files. (y/N): ").strip().lower()
    if confirm2 != "y":
        print("❌ Operation cancelled at second confirmation.")
        sys.exit(0)


def main():
    confirm_action()

    # clean unwanted files
    for folder in FOLDERS_TO_CLEAN:
        clean_folder(BASE / folder)

    # recreate/truncate required CSVs
    for rel, headers in CSV_HEADERS.items():
        truncate_or_create_csv(BASE / rel, headers)

    print("\n✅ Pipeline cleaned. CSV headers preserved (or re-created), all other data cleared.")


if __name__ == "__main__":
    main()
