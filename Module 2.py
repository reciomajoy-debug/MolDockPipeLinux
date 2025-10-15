#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 2 (RDKit): SMILES -> 3D SDF  [GRACEFUL STOP ENABLED]
- No OpenBabel; uses RDKit ETKDGv3 + UFF/MMFF and atomic writes.
- Inputs:
    - input/input.csv (id,smiles,notes,params_json)
    - state/admet_pass.list (optional; else all ligands)
    - config/run.yml, config/machine.yml (optional; chemistry/policy)
- Outputs (beside each other for transparency):
    - 3D_Structures/<id>.smi
    - 3D_Structures/<id>.sdf  (atomic write; never left empty)
    - 3D_Structures/<id>_rdkit.log  (overwritten each run)
    - Updates state/manifest.csv (sdf_* fields)

Run:  python Module 2.py
"""

from __future__ import annotations
import csv
import hashlib
import json
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone

# --- Graceful stop flags ---
STOP_REQUESTED = False
HARD_STOP = False

def _handle_sigint(sig, frame):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C detected — finishing current ligand, then exiting cleanly...")
        print("   (Press Ctrl+C again to stop immediately after a safe checkpoint.)")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will abort the loop ASAP and finalize outputs.")
signal.signal(signal.SIGINT, _handle_sigint)

# --- Required deps ---
from rdkit import Chem
from rdkit.Chem import AllChem

# Optional (for YAML configs)
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# ------------------------------ Paths ----------------------------------------
BASE = Path(".").resolve()
DIR_INPUT = BASE / "input"
DIR_STATE = BASE / "state"
DIR_SDF = BASE / "3D_Structures"
DIR_LOGS = BASE / "logs"

FILE_INPUT = DIR_INPUT / "input.csv"
FILE_PASS = DIR_STATE / "admet_pass.list"
FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_RUNYML = BASE / "config" / "run.yml"
FILE_MACHINEYML = BASE / "config" / "machine.yml"

for d in (DIR_INPUT, DIR_STATE, DIR_SDF, DIR_LOGS):
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

def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def normalize_id(row_id: str|None, smiles: str) -> str:
    rid = (row_id or "").strip()
    if rid:
        return rid
    return f"UNK_{hashlib.sha1(smiles.encode('utf-8')).hexdigest()[:10]}"

def log_write(path: Path, text: str, mode: str = "w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        f.write(text)

# ------------------------------ Config ---------------------------------------
DEFAULTS = {
    "chemistry": {
        "geometry_method": "ETKDGv3+UFF",  # info only
        "force_field": "UFF",               # UFF or MMFF
        "minimize_steps": 200
    },
    "policy": {
        "skip_if_done": True                # skip only if existing SDF validates
    }
}

def deep_update(dst: dict, src: dict):
    for k, v in src.items():
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
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    deep_update(cfg, load_yaml(FILE_RUNYML))
    deep_update(cfg, load_yaml(FILE_MACHINEYML))
    return cfg

def config_hash() -> str:
    chunks = []
    for p in (FILE_RUNYML, FILE_MACHINEYML):
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8"))
    if not chunks:
        chunks.append(json.dumps(DEFAULTS, sort_keys=True))
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

# ------------------------------ Validation -----------------------------------
def sdf_is_valid(path: Path) -> bool:
    """Strict validation: ensure file exists, non-trivial size, and RDKit can read ≥1 mol."""
    try:
        if not path.exists() or path.stat().st_size < 200:
            return False
        suppl = Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in suppl:
            if mol is not None:
                return True
        return False
    except Exception:
        return False

# ------------------------------ RDKit 3D builder -----------------------------
def rdkit_make_sdf(smiles: str, out_sdf: Path, ff: str = "UFF", max_iters: int = 200) -> tuple[bool, str]:
    """
    Generate 3D using RDKit ETKDGv3 + UFF/MMFF with ATOMIC WRITE.
    Returns (ok, reason). Never leaves an empty final SDF.
    """
    out_sdf = out_sdf.resolve()
    out_sdf.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale final file to avoid reusing bad/locked outputs
    try:
        if out_sdf.exists():
            out_sdf.unlink()
    except Exception:
        pass

    tmp_sdf = out_sdf.with_suffix(".sdf.tmp")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit: invalid SMILES"

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        code = AllChem.EmbedMolecule(mol, params=params)
        if code != 0:
            return False, "RDKit: ETKDG embed failed"

        # Force-field minimization
        ff = (ff or "UFF").upper()
        if ff.startswith("MMFF"):
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            if props is None:
                ff = "UFF"
            else:
                ffobj = AllChem.MMFFGetMoleculeForceField(mol, props)
                ffobj.Initialize()
                ffobj.Minimize(maxIts=int(max_iters))
        if ff == "UFF":
            ffobj = AllChem.UFFGetMoleculeForceField(mol)
            ffobj.Initialize()
            ffobj.Minimize(maxIts=int(max_iters))

        # Atomic write to tmp
        w = Chem.SDWriter(str(tmp_sdf))
        w.write(mol)
        w.close()

        # Validate tmp; if OK → rename into place
        if not sdf_is_valid(tmp_sdf):
            try:
                tmp_sdf.unlink()
            except Exception:
                pass
            return False, "RDKit: wrote SDF but validation failed"

        tmp_sdf.replace(out_sdf)  # atomic on same filesystem
        return True, "OK"
    except Exception as e:
        try:
            if tmp_sdf.exists():
                tmp_sdf.unlink()
        except Exception:
            pass
        return False, f"RDKit error: {e}"

# ------------------------------ Main -----------------------------------------
def main():
    if not FILE_INPUT.exists():
        raise SystemExit("❌ Missing input/input.csv")

    cfg = load_config()
    chash = config_hash()
    created_ts = now_iso()

    minimize_steps = int(cfg["chemistry"].get("minimize_steps", 200))
    force_field = str(cfg["chemistry"].get("force_field", "UFF")).upper()
    skip_if_done = bool(cfg["policy"].get("skip_if_done", True))

    # Determine ligands to process
    input_rows = read_csv(FILE_INPUT)
    if not input_rows:
        raise SystemExit("⚠️ input/input.csv has no rows.")

    id2smiles = {}
    for r in input_rows:
        smiles = (r.get("smiles") or "").strip()
        if not smiles:
            continue
        lig_id = normalize_id(r.get("id"), smiles)
        id2smiles[lig_id] = smiles

    ids = read_lines(FILE_PASS)
    if not ids:
        ids = list(id2smiles.keys())

    # Manifest
    manifest: dict[str, dict] = load_manifest()
    done, failed = 0, 0

    try:
        for idx, lig_id in enumerate(ids, 1):
            # Respect user stop request
            if STOP_REQUESTED:
                print("🧾 Stop requested — finalizing after this checkpoint...")
                break
            if HARD_STOP:
                print("🧾 Hard stop — exiting loop immediately after checkpoint...")
                break

            smiles = id2smiles.get(lig_id, "")
            out_sdf = (DIR_SDF / f"{lig_id}.sdf").resolve()
            out_smi = out_sdf.with_suffix(".smi")
            out_log = out_sdf.with_name(out_sdf.stem + "_rdkit.log")

            # Always overwrite .smi for transparency
            out_smi.parent.mkdir(parents=True, exist_ok=True)
            out_smi.write_text(smiles + "\n", encoding="utf-8")

            # Skip only if existing SDF validates
            if skip_if_done and sdf_is_valid(out_sdf):
                m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
                m["id"] = lig_id
                m["smiles"] = smiles
                m.setdefault("created_at", created_ts)
                m["updated_at"] = now_iso()
                m["sdf_status"] = "DONE"
                m["sdf_path"] = str(out_sdf)
                m["sdf_reason"] = "Found existing valid SDF"
                m["config_hash"] = chash
                # Record RDKit version
                m["tools_rdkit"] = getattr(Chem, "__version__", "RDKit")
                manifest[lig_id] = m
                done += 1
                log_write(out_log, f"[SKIP] Existing valid SDF kept: {out_sdf}\n")
                # light checkpoint: write every 50 ligs
                if idx % 50 == 0:
                    save_manifest(manifest)
                continue

            # Fresh RDKit build (atomic)
            ok, reason = rdkit_make_sdf(smiles, out_sdf, ff=force_field, max_iters=minimize_steps)

            # Atom count (for log)
            atom_count = "?"
            try:
                if ok:
                    suppl = Chem.SDMolSupplier(str(out_sdf), removeHs=False)
                    mol0 = next((m for m in suppl if m is not None), None)
                    if mol0:
                        atom_count = str(mol0.GetNumAtoms())
            except Exception:
                pass

            # Overwrite log each run
            log_text = [
                f"RDKit build for {lig_id}",
                f"SMILES: {smiles}",
                f"Force field: {force_field}",
                f"Minimize steps: {minimize_steps}",
                f"Output SDF: {out_sdf}",
                f"Result: {'OK' if ok else 'FAIL'}; Reason: {reason}; atoms={atom_count}",
            ]
            log_write(out_log, "\n".join(log_text) + "\n", mode="w")

            # Update manifest
            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["smiles"] = smiles
            if not m.get("created_at"):
                m["created_at"] = created_ts
            m["updated_at"] = now_iso()
            m["config_hash"] = chash
            m["sdf_path"] = str(out_sdf)
            m["sdf_status"] = "DONE" if ok else "FAILED"
            m["sdf_reason"] = "OK" if ok else reason
            m["tools_rdkit"] = getattr(Chem, "__version__", "RDKit")
            manifest[lig_id] = m

            if ok:
                done += 1
            else:
                failed += 1

            # periodic checkpoint to be extra safe
            if idx % 50 == 0:
                save_manifest(manifest)

    finally:
        # Always flush manifest even on Ctrl+C or unexpected exceptions
        save_manifest(manifest)
        print(f"✅ RDKit SMILES→SDF complete (or stopped). DONE: {done}  FAILED: {failed}")
        print(f"   Outputs in: {DIR_SDF}")
        print(f"   Manifest updated: {FILE_MANIFEST}")
        if STOP_REQUESTED or HARD_STOP:
            print("   (Exited early by user request.)")

if __name__ == "__main__":
    main()
