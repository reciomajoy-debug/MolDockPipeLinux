#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1: ADMET Screening
- Reads input/input.csv (id, smiles, ...)
- Computes descriptors via RDKit (if available) and applies rules (Lipinski/Veber/Egan/Ghose)
- Writes output/admet.csv, state/admet_pass.list, state/admet_fail.list
- Updates state/manifest.csv with ADMET status/reason and InChIKey
- Idempotent: merges into manifest; safe to re-run

Run:  python m1_admet.py
"""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

# ----------------------- Optional deps (graceful) -----------------------------
try:
    import yaml  # PyYAML (optional). If missing, we use defaults.
except Exception:
    yaml = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# ------------------------------ Paths ----------------------------------------
BASE = Path(".").resolve()
DIR_INPUT = BASE / "input"
DIR_OUTPUT = BASE / "output"
DIR_STATE = BASE / "state"
DIR_LOGS = BASE / "logs"

FILE_INPUT = DIR_INPUT / "input.csv"
FILE_ADMET = DIR_OUTPUT / "admet.csv"
FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_PASS = DIR_STATE / "admet_pass.list"
FILE_FAIL = DIR_STATE / "admet_fail.list"
FILE_RUNYML = BASE / "config" / "run.yml"

for d in (DIR_INPUT, DIR_OUTPUT, DIR_STATE, DIR_LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------ Config ---------------------------------------
DEFAULT_CONFIG = {
    "admet_rules": {
        "lipinski": True,
        "veber": True,
        "egan": False,
        "ghose": False,
        "hard_fail": False  # if True, fail any rule violation; if False, record but still PASS unless Lipinski/Veber fail
    }
}

def load_run_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if FILE_RUNYML.exists() and yaml is not None:
        try:
            data = yaml.safe_load(FILE_RUNYML.read_text(encoding="utf-8")) or {}
            # Deep-merge only for admet_rules we care about
            if isinstance(data, dict) and "admet_rules" in data and isinstance(data["admet_rules"], dict):
                cfg["admet_rules"].update({k: bool(v) for k, v in data["admet_rules"].items()})
        except Exception:
            pass
    return cfg

def config_hash() -> str:
    if FILE_RUNYML.exists():
        try:
            text = FILE_RUNYML.read_text(encoding="utf-8")
            return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        except Exception:
            pass
    # fallback to defaults hash
    return hashlib.sha1(json.dumps(DEFAULT_CONFIG, sort_keys=True).encode("utf-8")).hexdigest()[:10]

# ------------------------------ IO helpers -----------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def safe_csv_write(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

def read_csv_as_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def normalize_id(row_id: str|None, smiles: str) -> str:
    rid = (row_id or "").strip()
    if rid:
        return rid
    return f"UNK_{hashlib.sha1(smiles.encode('utf-8')).hexdigest()[:10]}"

# ------------------------------ Rules/Descriptors ----------------------------
def compute_descriptors(smiles: str):
    """Return dict of descriptors or None if RDKit unavailable/invalid."""
    if not RDKit_OK:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    # descriptors
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    inchikey = Chem.MolToInchiKey(mol)
    return {
        "mw": round(mw, 2),
        "alogp": round(clogp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": int(hbd),
        "hba": int(hba),
        "rotb": int(rotb),
        "inchikey": inchikey
    }

def apply_rules(desc: dict, rules_cfg: dict) -> tuple[str, str, dict]:
    """
    Returns (decision, reason, flags)
    decision: PASS | FAIL | SKIPPED
    reason: short string
    flags: per-rule booleans (lipinski_ok, ...)
    """
    # default flags
    flags = {"lipinski_ok": "", "veber_ok": "", "egan_ok": "", "ghose_ok": ""}

    # Without RDKit, we can't compute; mark SKIPPED
    if desc is None:
        return "SKIPPED", "RDKit not available", flags
    if desc == "INVALID":
        return "FAIL", "Invalid SMILES", flags

    mw, clogp, tpsa = desc["mw"], desc["alogp"], desc["tpsa"]
    hbd, hba, rotb = desc["hbd"], desc["hba"], desc["rotb"]

    # Evaluate rules (typical thresholds)
    lipinski_ok = (mw <= 500) and (clogp <= 5) and (hbd <= 5) and (hba <= 10)
    veber_ok = (tpsa <= 140) and (rotb <= 10)
    egan_ok = (clogp <= 5.88) and (tpsa <= 131)
    ghose_ok = (160 <= mw <= 480) and (-0.4 <= clogp <= 5.6)

    flags.update({
        "lipinski_ok": str(bool(lipinski_ok)).upper(),
        "veber_ok": str(bool(veber_ok)).upper(),
        "egan_ok": str(bool(egan_ok)).upper(),
        "ghose_ok": str(bool(ghose_ok)).upper(),
    })

    active = rules_cfg or {}
    hard_fail = bool(active.get("hard_fail", False))

    violations = []
    if active.get("lipinski", True) and not lipinski_ok:
        violations.append("Lipinski")
    if active.get("veber", True) and not veber_ok:
        violations.append("Veber")
    if active.get("egan", False) and not egan_ok:
        violations.append("Egan")
    if active.get("ghose", False) and not ghose_ok:
        violations.append("Ghose")

    if not violations:
        return "PASS", "All rules satisfied", flags

    # If there are violations
    if hard_fail:
        return "FAIL", "; ".join(v + " violated" for v in violations), flags

    # Default: only Lipinski/Veber violations cause FAIL, others warn but pass
    must_fail = any(v in ("Lipinski", "Veber") for v in violations)
    if must_fail:
        return "FAIL", "; ".join(v + " violated" for v in violations), flags
    else:
        return "PASS", "; ".join(v + " violated (allowed)" for v in violations), flags

# ------------------------------ Manifest -------------------------------------
MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_obabel","tools_meeko","tools_vina",
    "created_at","updated_at"
]

def load_manifest(path: Path) -> dict[str, dict]:
    """Return dict keyed by id with full manifest row (missing fields empty)."""
    existing = {}
    if not path.exists():
        return existing
    rows = read_csv_as_dicts(path)
    for r in rows:
        row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
        existing[row["id"]] = row
    return existing

def save_manifest(path: Path, data: dict[str, dict]) -> None:
    rows = []
    for k in sorted(data.keys()):
        # ensure all fields present
        row = {key: data[k].get(key, "") for key in MANIFEST_FIELDS}
        rows.append(row)
    safe_csv_write(path, rows, MANIFEST_FIELDS)

# ------------------------------ Main -----------------------------------------
def main():
    # Sanity: input exists
    if not FILE_INPUT.exists():
        raise SystemExit("❌ Missing input/input.csv (headers: id,smiles,notes,params_json)")

    cfg = load_run_config()
    chash = config_hash()
    created_ts = now_iso()

    # Ingest input
    input_rows = read_csv_as_dicts(FILE_INPUT)
    if not input_rows:
        print("⚠️ input/input.csv is empty (only headers?) — nothing to do.")
        # still create empty outputs with headers
        safe_csv_write(FILE_ADMET, [], [
            "id","smiles","inchikey","mw","alogp","tpsa","hbd","hba","rotb",
            "lipinski_ok","veber_ok","egan_ok","ghose_ok",
            "admet_decision","admet_reason","config_hash","tools_rdkit","created_at"
        ])
        write_lines(FILE_PASS, [])
        write_lines(FILE_FAIL, [])
        # manifest (headers only if not present)
        if not FILE_MANIFEST.exists():
            safe_csv_write(FILE_MANIFEST, [], MANIFEST_FIELDS)
        return

    # Load manifest so we can merge/update
    manifest = load_manifest(FILE_MANIFEST)

    # Prepare outputs
    admet_rows = []
    pass_ids = []
    fail_ids = []

    tools_rdkit = "RDKit available" if RDKit_OK else "RDKit not available"

    for raw in input_rows:
        smiles = (raw.get("smiles") or "").strip()
        if not smiles:
            # skip empty smiles rows
            continue
        lig_id = normalize_id(raw.get("id"), smiles)

        # Compute descriptors
        desc = compute_descriptors(smiles)
        # Apply rules
        decision, reason, flags = apply_rules(desc, cfg.get("admet_rules", {}))

        # Compose ADMET row
        admet_row = {
            "id": lig_id,
            "smiles": smiles,
            "inchikey": "" if (desc in (None, "INVALID")) else desc["inchikey"],
            "mw": "" if (desc in (None, "INVALID")) else desc["mw"],
            "alogp": "" if (desc in (None, "INVALID")) else desc["alogp"],
            "tpsa": "" if (desc in (None, "INVALID")) else desc["tpsa"],
            "hbd": "" if (desc in (None, "INVALID")) else desc["hbd"],
            "hba": "" if (desc in (None, "INVALID")) else desc["hba"],
            "rotb": "" if (desc in (None, "INVALID")) else desc["rotb"],
            "lipinski_ok": flags["lipinski_ok"],
            "veber_ok": flags["veber_ok"],
            "egan_ok": flags["egan_ok"],
            "ghose_ok": flags["ghose_ok"],
            "admet_decision": "SKIPPED" if decision == "SKIPPED" else decision,
            "admet_reason": reason,
            "config_hash": chash,
            "tools_rdkit": tools_rdkit,
            "created_at": created_ts
        }
        admet_rows.append(admet_row)

        # Side lists
        if decision == "FAIL":
            fail_ids.append(lig_id)
        elif decision in ("PASS", "SKIPPED"):
            pass_ids.append(lig_id)

        # Merge into manifest
        m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
        # fill identity
        m["id"] = lig_id
        m["smiles"] = smiles
        if admet_row["inchikey"]:
            m["inchikey"] = admet_row["inchikey"]
        # ADMET status
        status_map = {
            "PASS": "PASSED",
            "FAIL": "FAILED",
            "SKIPPED": "SKIPPED_ADMET"
        }
        m["admet_status"] = status_map.get(admet_row["admet_decision"], "PENDING")
        m["admet_reason"] = reason

        # Touch timestamps/config
        if not m.get("created_at"):
            m["created_at"] = created_ts
        m["updated_at"] = now_iso()
        m["config_hash"] = chash

        manifest[lig_id] = m

    # Write outputs
    safe_csv_write(FILE_ADMET, admet_rows, [
        "id","smiles","inchikey","mw","alogp","tpsa","hbd","hba","rotb",
        "lipinski_ok","veber_ok","egan_ok","ghose_ok",
        "admet_decision","admet_reason","config_hash","tools_rdkit","created_at"
    ])
    write_lines(FILE_PASS, pass_ids)
    write_lines(FILE_FAIL, fail_ids)
    save_manifest(FILE_MANIFEST, manifest)

    # Operator summary
    print(f"✅ ADMET done. Rows: {len(admet_rows)}  PASS: {len(pass_ids)}  FAIL: {len(fail_ids)}")
    print(f"   Wrote: {FILE_ADMET}")
    print(f"   Lists: {FILE_PASS.name} ({len(pass_ids)}) , {FILE_FAIL.name} ({len(fail_ids)})")
    print(f"   Manifest updated: {FILE_MANIFEST}")

if __name__ == "__main__":
    main()
