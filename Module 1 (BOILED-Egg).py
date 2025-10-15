#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1 (with BOILED-Egg YOLK gate): ADMET Screening

- Reads input/input.csv (expects at least: id,smiles)
- Computes descriptors via RDKit (if available) and applies rules (Lipinski, Veber)
- Adds BOILED-Egg classification using RDKit's WLOGP (Crippen.MolLogP) and TPSA
  YOLK (BBB-likely) is defined pragmatically per Daina & Zoete (2016) text:
      TPSA < 79 Å² and 0.4 ≤ WLOGP ≤ 6.0  -> YOLK
  (We focus on YOLK because the user requires YOLK to pass Module 1.)
- Writes output/admet.csv and state/{admet_pass.list, admet_fail.list}
- Merges/updates state/manifest.csv with per-id status and InChIKey
- Idempotent and safe to re-run; makes folders if missing.

Run:  python "Module 1 (BOILED-Egg).py"
"""

from __future__ import annotations
import csv, sys, os, json, time, hashlib
from pathlib import Path
from typing import Dict, Any, Tuple

# -------- Paths --------
ROOT = Path(__file__).resolve().parent
DIR_INPUT   = ROOT / "input"
DIR_OUTPUT  = ROOT / "output"
DIR_STATE   = ROOT / "state"

FILE_INPUT     = DIR_INPUT / "input.csv"
FILE_ADMET     = DIR_OUTPUT / "admet.csv"
FILE_PASS      = DIR_STATE / "admet_pass.list"
FILE_FAIL      = DIR_STATE / "admet_fail.list"
FILE_MANIFEST  = DIR_STATE / "manifest.csv"

# -------- Optional deps (RDKit) --------
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors, Descriptors, Lipinski
except Exception as e:
    Chem = None  # type: ignore
    Crippen = rdMolDescriptors = Descriptors = Lipinski = None  # type: ignore

def sha1_of_text(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def ensure_dirs():
    for d in (DIR_INPUT, DIR_OUTPUT, DIR_STATE):
        d.mkdir(parents=True, exist_ok=True)

def read_input_rows() -> list[dict]:
    if not FILE_INPUT.exists():
        raise FileNotFoundError(f"Missing {FILE_INPUT}. Create it with headers: id,smiles,...")
    rows: list[dict] = []
    with FILE_INPUT.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'id' not in reader.fieldnames or 'smiles' not in reader.fieldnames:
            raise ValueError("input.csv must contain at least 'id' and 'smiles' headers.")
        for r in reader:
            if not (r.get('id') and r.get('smiles')):
                continue
            rows.append({'id': r['id'].strip(), 'smiles': r['smiles'].strip()})
    return rows

# -------- Descriptor & Rules --------
def compute_descriptors(smiles: str) -> Tuple[Dict[str, Any], str]:
    """Return (desc, inchi_key). If RDKit missing or mol invalid, return minimal info."""
    if Chem is None:
        return ({
            'valid': False, 'error': 'rdkit_not_available',
            'mw': None, 'hbd': None, 'hba': None, 'rotb': None,
            'tpsa': None, 'wlogp': None, 'rings': None
        }, '')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ({'valid': False, 'error': 'invalid_smiles',
                 'mw': None, 'hbd': None, 'hba': None, 'rotb': None,
                 'tpsa': None, 'wlogp': None, 'rings': None}, '')
    try:
        mw    = Descriptors.ExactMolWt(mol)
        hbd   = Lipinski.NumHDonors(mol)
        hba   = Lipinski.NumHAcceptors(mol)
        rotb  = Lipinski.NumRotatableBonds(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)
        wlogp = Crippen.MolLogP(mol)  # Wildman–Crippen
        rings = Lipinski.RingCount(mol)
        inchi = Chem.MolToInchiKey(mol)
        return ({
            'valid': True, 'mw': mw, 'hbd': hbd, 'hba': hba, 'rotb': rotb,
            'tpsa': tpsa, 'wlogp': wlogp, 'rings': rings,
        }, inchi)
    except Exception as e:
        return ({'valid': False, 'error': f'rdkit_error:{e}'}, '')

def rule_lipinski(d: Dict[str, Any]) -> Tuple[bool, str]:
    if not d.get('valid'): return (False, 'invalid')
    ok = (d['mw'] <= 500 + 1e-9 and d['hbd'] <= 5 and d['hba'] <= 10 and d['wlogp'] <= 5 + 1e-9)
    return (ok, '' if ok else 'lipinski')

def rule_veber(d: Dict[str, Any]) -> Tuple[bool, str]:
    if not d.get('valid'): return (False, 'invalid')
    ok = (d['tpsa'] <= 140 + 1e-9 and d['rotb'] <= 10)
    return (ok, '' if ok else 'veber')

def boiled_egg_region(d: Dict[str, Any]) -> str:
    """Return 'YOLK' if BBB-likely; else 'WHITE' or 'GREY' based on TPSA/WLOGP.
    We prioritize YOLK decision using thresholds reported in Daina & Zoete (2016).
    """
    if not d.get('valid'): return 'GREY'
    tpsa = float(d['tpsa']); wlogp = float(d['wlogp'])
    yolk = (tpsa < 79.0 and 0.4 <= wlogp <= 6.0)
    if yolk: return 'YOLK'
    # simple "white" heuristic (high HIA) using Egan-like polarity/lipophilicity window
    white = (tpsa <= 130.0 and -0.4 <= wlogp <= 5.6)
    return 'WHITE' if white else 'GREY'

def decide_pass(d: Dict[str, Any], require_yolk: bool = True) -> Tuple[str, str]:
    """Return (decision, reason). 'PASS' only if Lipinski, Veber, and YOLK if configured."""
    if not d.get('valid'):
        return ('FAIL', 'invalid')
    lip, r1 = rule_lipinski(d)
    veb, r2 = rule_veber(d)
    egg = boiled_egg_region(d)
    if not lip: return ('FAIL', r1)
    if not veb: return ('FAIL', r2)
    if require_yolk and egg != 'YOLK': return ('FAIL', 'boiled_egg:not_yolk')
    return ('PASS', 'ok')

def read_manifest() -> Dict[str, Dict[str, str]]:
    m: Dict[str, Dict[str, str]] = {}
    if FILE_MANIFEST.exists():
        with FILE_MANIFEST.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                m[r['id']] = r
    return m

def write_manifest(updated: Dict[str, Dict[str, str]]):
    fields = ['id','inchi_key','admet_decision','admet_reason','created_at']
    with FILE_MANIFEST.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for _id in sorted(updated.keys()):
            w.writerow({k: updated[_id].get(k,'') for k in fields})

def write_list(path: Path, ids: list[str]):
    with path.open('w', encoding='utf-8') as f:
        for i in ids:
            f.write(i+'\n')

def main():
    ensure_dirs()
    rows = read_input_rows()
    created_at = time.strftime('%Y-%m-%d %H:%M:%S')
    out_rows = []
    pass_ids, fail_ids = [], []
    manifest = read_manifest()

    for r in rows:
        desc, inchi = compute_descriptors(r['smiles'])
        egg = boiled_egg_region(desc) if desc.get('valid') else 'GREY'
        decision, reason = decide_pass(desc, require_yolk=True)
        rec = {
            'id': r['id'],
            'smiles': r['smiles'],
            'inchi_key': inchi,
            'mw': desc.get('mw'),
            'hbd': desc.get('hbd'),
            'hba': desc.get('hba'),
            'rotb': desc.get('rotb'),
            'tpsa': desc.get('tpsa'),
            'wlogp': desc.get('wlogp'),
            'rings': desc.get('rings'),
            'boiled_egg': egg,
            'admet_decision': decision,
            'admet_reason': reason,
        }
        out_rows.append(rec)
        (pass_ids if decision=='PASS' else fail_ids).append(r['id'])
        manifest[r['id']] = {
            'id': r['id'],
            'inchi_key': inchi,
            'admet_decision': decision,
            'admet_reason': reason,
            'created_at': created_at
        }

    # write outputs
    with FILE_ADMET.open('w', newline='', encoding='utf-8') as f:
        fields = ['id','smiles','inchi_key','mw','hbd','hba','rotb','tpsa','wlogp','rings','boiled_egg','admet_decision','admet_reason']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    write_list(FILE_PASS, pass_ids)
    write_list(FILE_FAIL, fail_ids)
    write_manifest(manifest)

    print(f"✅ ADMET done. Rows={len(out_rows)} PASS={len(pass_ids)} FAIL={len(fail_ids)}")
    print(f"   Wrote: {FILE_ADMET}")
    print(f"   Lists: {FILE_PASS.name} ({len(pass_ids)}), {FILE_FAIL.name} ({len(fail_ids)})")
    print(f"   Manifest updated: {FILE_MANIFEST}")

if __name__ == "__main__":
    main()
