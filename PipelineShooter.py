#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Molecular Docking Pipeline Runner (macOS / Linux / Windows)
Lets the user manually choose which module file to run for each stage.

Usage:
    python run_pipeline_selective.py
"""

import subprocess
import sys
import shlex
from pathlib import Path

BASE = Path(__file__).resolve().parent

stages = [
    ("Module 1 (ADMET Screening)", "module_1"),
    ("Module 2 (3D Structure Generation)", "module_2"),
    ("Module 3 (Ligand Preparation)", "module_3"),
    ("Module 4 (Docking GPU)", "module_4"),
]

def prompt_for_module(stage_name):
    print(f"\n📦 {stage_name}")
    print(f"   Files available in {BASE}:")
    all_scripts = sorted(BASE.glob("Module*.py"))
    for f in all_scripts:
        print(f"     - {f.name}")
    sel = input(f"👉 Enter filename to use for {stage_name} (or leave blank to skip): ").strip()
    if not sel:
        return None
    script_path = BASE / sel
    if not script_path.exists():
        print(f"❌ File '{sel}' not found. Skipping this stage.")
        return None
    return script_path

def run_module(script_path: Path, stage_name: str):
    print(f"\n🚀 Running {stage_name}: {script_path.name}")
    print("=" * 80)
    # Universal Python launcher (works on Mac/Linux/Win)
    cmd = [sys.executable, str(script_path)]
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user. Exiting cleanly.")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error running {script_path.name}: {e}")
        sys.exit(1)

    if rc != 0:
        print(f"⚠️ {stage_name} exited with code {rc}. Pipeline stopping.")
        sys.exit(rc)
    print(f"✅ {stage_name} completed.\n")

def main():
    print("🧬 Molecular Docking Pipeline — Cross-Platform Selective Runner")
    print("Working directory:", BASE)
    print("=" * 80)

    chosen_modules = []
    for stage_name, _ in stages:
        m = prompt_for_module(stage_name)
        if m:
            chosen_modules.append((stage_name, m))

    if not chosen_modules:
        print("❌ No modules selected. Exiting.")
        sys.exit(0)

    print("\n🧩 Selected modules:")
    for name, path in chosen_modules:
        print(f"  - {name}: {path.name}")

    confirm = input("\nProceed with these modules? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    for name, path in chosen_modules:
        run_module(path, name)

    print("🎉 All selected modules completed successfully!")

if __name__ == "__main__":
    main()
