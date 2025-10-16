#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4c (Linux, QuickVina 2.1): Idempotent CPU docking with graceful stop
- Mirrors Module 4a behavior but uses QuickVina (qvina2.1/qvina2/qvina)
- Inputs:
    prepared_ligands/*.pdbqt    (from Module 3)
    VinaConfig.txt              (box + params: receptor, center_x/y/z, size_x/y/z, exhaustiveness, num_modes, energy_range)
- Outputs:
    results/<id>_out.pdbqt
    results/<id>_qvina.log
    state/manifest.csv          (vina_* fields updated)
    results/{summary.csv, leaderboard.csv}
- Idempotent:
    Skips ligands that already have valid *_out.pdbqt with a Vina score
- Graceful stop:
    1× Ctrl+C = finish current ligand & checkpoint then exit cleanly
    2× Ctrl+C = stop ASAP after safe checkpoint
    STOP file in repo root also requests a clean stop
- Atomic writes for pose/log; manifest & summaries flushed periodically

Run:
    python3 "Module 4c (QVina Linux).py"

Binary discovery order:
    $QVINA_BIN (env) →
    shutil.which("qvina2.1") →
    shutil.which("qvina2") →
    shutil.which("qvina")

Tested on Linux; Windows/macOS not targeted in this module (use 4a/4b).
"""

from __future__ import annotations
import csv, hashlib, os, re, shlex, shutil, signal, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Iterable

# ---------------------------- Paths & constants ------------------------------
BASE = Path(".").resolve()
DIR_PREP     = BASE / "prepared_ligands"
DIR_RESULTS  = BASE / "results"
DIR_STATE    = BASE / "state"
FILE_MANIFEST= DIR_STATE / "manifest.csv"
FILE_SUMMARY = DIR_RESULTS / "summary.csv"
FILE_LEADER  = DIR_RESULTS / "leaderboard.csv"
FILE_CFG     = BASE / "VinaConfig.txt"     # reused for QuickVina
FILE_STOP    = BASE / "STOP"               # graceful stop flag

for d in (DIR_RESULTS, DIR_STATE):
    d.mkdir(parents=True, exist_ok=True)

MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]

# Accept typical QuickVina/Vina style “REMARK VINA RESULT: -x.xx”
RES_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s+(-?\d+\.\d+)", re.I)

# ---------------------------- Graceful stop ----------------------------------
STOP_REQUESTED = False
HARD_STOP = False

def _sigint(sig, frame):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C — finishing this ligand then exiting cleanly…")
        print("    (Press Ctrl+C again to stop ASAP after checkpoint.)")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will exit ASAP after checkpoint.")
signal.signal(signal.SIGINT, _sigint)

def stop_file_requested() -> bool:
    try:
        return FILE_STOP.exists()
    except Exception:
        return False

# ---------------------------- Small utils ------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

def sha1_of_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest() -> dict[str, dict]:
    if not FILE_MANIFEST.exists(): return {}
    out={}
    for r in read_csv(FILE_MANIFEST):
        row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
        out[row["id"]] = row
    return out

def save_manifest(m: dict[str, dict]) -> None:
    rows = [{k: v.get(k, "") for k in MANIFEST_FIELDS} for _, v in sorted(m.items())]
    write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# ---------------------------- Config parsing ---------------------------------
def parse_cfg(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"❌ Config not found: {path} (expected VinaConfig.txt with receptor/box/params)")
    conf={}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if "#" in line: line = line.split("#",1)[0].strip()
        if "=" not in line: continue
        k, v = line.split("=", 1)
        conf[k.strip().lower()] = v.strip()
    # Normalize a few aliases
    if "receptor_file" in conf and "receptor" not in conf:
        conf["receptor"] = conf["receptor_file"]
    return conf

def as_float(d: Dict[str, str], k: str, default: float) -> float:
    try: return float(d.get(k, default))
    except: return float(default)

def as_int(d: Dict[str, str], k: str, default: int) -> int:
    try: return int(str(d.get(k, default)).strip())
    except: return int(default)

def config_hash(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        txt = "{}"
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:10]

# ---------------------------- QVina binary detection -------------------------
def find_qvina_binary() -> Path:
    """
    Locate QuickVina executable in this order:
    1. Current working directory (same folder as script)
    2. $QVINA_BIN or $VINA_BIN environment variable
    3. System PATH via shutil.which()
    """
    local_names = ("qvina2.1", "qvina2", "qvina")
    for name in local_names:
        candidate = Path(name)
        if candidate.exists():
            return candidate.resolve()

    env = os.environ.get("QVINA_BIN") or os.environ.get("VINA_BIN")
    if env:
        p = Path(env)
        if p.exists():
            return p.resolve()

    for name in local_names:
        exe = shutil.which(name)
        if exe:
            return Path(exe).resolve()

    raise SystemExit(
        "❌ QuickVina binary not found. "
        "Place qvina2.1/qvina2/qvina in the same folder as this script "
        "or set $QVINA_BIN to its path."
    )

# ---------------------------- Pose/score validation --------------------------
def pose_is_valid(pose_path: Path) -> Tuple[bool, Optional[float]]:
    try:
        if not pose_path.exists() or pose_path.stat().st_size < 200:
            return (False, None)
        txt = pose_path.read_text(errors="ignore")
        scores = [float(m.group(1)) for m in RES_RE.finditer(txt)]
        return ((len(scores) > 0), (min(scores) if scores else None))
    except Exception:
        return (False, None)

# ---------------------------- Summaries --------------------------------------
def build_and_write_summaries(manifest: dict[str, dict]) -> None:
    summ_headers = ["id","inchikey","vina_score","pose_path","created_at"]
    rows=[]
    for _, m in sorted(manifest.items()):
        sc = m.get("vina_score","")
        if sc:
            rows.append({
                "id": m.get("id",""),
                "inchikey": m.get("inchikey",""),
                "vina_score": sc,
                "pose_path": m.get("vina_pose",""),
                "created_at": m.get("updated_at","")
            })
    write_csv(FILE_SUMMARY, rows, summ_headers)

    lead_headers = ["rank","id","inchikey","vina_score","pose_path"]
    ranked = sorted(rows, key=lambda r: float(r["vina_score"])) if rows else []
    leaders = [
        {"rank": i, "id": r["id"], "inchikey": r["inchikey"],
         "vina_score": r["vina_score"], "pose_path": r["pose_path"]}
        for i, r in enumerate(ranked, 1)
    ]
    write_csv(FILE_LEADER, leaders, lead_headers)

# ---------------------------- Docking runner ---------------------------------
def run_qvina(qvina: Path, receptor: Path, ligand: Path, out_pose: Path, out_log: Path,
              box: dict, params: dict, quiet: bool = True) -> int:
    """
    Call QuickVina with typical flags compatible with VinaConfig.txt.
    Writes to tmp files first, then renames on success.
    Returns process return code (0 = success).
    """
    out_pose_tmp = out_pose.with_suffix(".pdbqt.tmp")
    out_log_tmp  = out_log.with_suffix(".log.tmp")

    # Clean any stale tmp/final (we will write atomically)
    for p in (out_pose_tmp, out_log_tmp):
        try:
            if p.exists(): p.unlink()
        except Exception:
            pass

    cmd = [
        str(qvina),
        "--receptor", str(receptor),
        "--ligand",   str(ligand),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x",   str(box["size_x"]),
        "--size_y",   str(box["size_y"]),
        "--size_z",   str(box["size_z"]),
        "--exhaustiveness", str(params["exhaustiveness"]),
        "--num_modes",      str(params["num_modes"]),
        "--energy_range",   str(params["energy_range"]),
        "--out",            str(out_pose_tmp),
        "--log",            str(out_log_tmp),
    ]

    if not quiet:
        print("QVina CMD:", " ".join(shlex.quote(c) for c in cmd))

    rc = subprocess.call(cmd, stdout=(subprocess.DEVNULL if quiet else None),
                              stderr=(subprocess.DEVNULL if quiet else None))
    # If ok, rename tmp → final
    if rc == 0:
        try:
            if out_pose_tmp.exists():
                out_pose_tmp.replace(out_pose)
            if out_log_tmp.exists():
                out_log_tmp.replace(out_log)
        except Exception:
            rc = 2  # treat atomic move failure as failure
    else:
        # cleanup tmp
        try:
            if out_pose_tmp.exists(): out_pose_tmp.unlink()
            if out_log_tmp.exists():  out_log_tmp.unlink()
        except Exception:
            pass
    return rc

# ---------------------------- Main -------------------------------------------
def main():
    if not DIR_PREP.exists():
        raise SystemExit("❌ Missing prepared_ligands/ (run Module 3 first)")

    qvina = find_qvina_binary()
    conf  = parse_cfg(FILE_CFG)

    receptor = Path(conf.get("receptor","")).expanduser()
    if not receptor.is_absolute():
        receptor = (BASE / receptor).resolve()
    if not receptor.exists():
        raise SystemExit(f"❌ Receptor not found: {receptor}")

    # --- Receptor: required, must be a file ---
    rec_cfg = conf.get("receptor", "").strip()
    if not rec_cfg:
        raise SystemExit("❌ VinaConfig.txt is missing 'receptor=...' (path to receptor PDBQT).")

    receptor = Path(rec_cfg).expanduser()
    if not receptor.is_absolute():
        receptor = (BASE / receptor).resolve()

    if receptor.is_dir():
        raise SystemExit(f"❌ 'receptor' points to a directory, not a file: {receptor}\n"
                         "   Set receptor=/path/to/receptor.pdbqt in VinaConfig.txt")

    if not receptor.exists():
        raise SystemExit(f"❌ Receptor file not found: {receptor}\n"
                         "   Check the path in VinaConfig.txt (receptor=...)")

    if receptor.suffix.lower() != ".pdbqt":
        print(f"⚠️  Receptor file does not end with .pdbqt: {receptor} (continuing)")

    box = {
        "center_x": as_float(conf, "center_x", 0.0),
        "center_y": as_float(conf, "center_y", 0.0),
        "center_z": as_float(conf, "center_z", 0.0),
        "size_x":   as_float(conf, "size_x", 20.0),
        "size_y":   as_float(conf, "size_y", 20.0),
        "size_z":   as_float(conf, "size_z", 20.0),
    }
    params = {
        "exhaustiveness": as_int(conf, "exhaustiveness", 8),
        "num_modes":      as_int(conf, "num_modes", 9),
        "energy_range":   as_int(conf, "energy_range", 3),
    }

    chash = config_hash(FILE_CFG)
    receptor_sha = sha1_of_file(receptor)
    

    # Discover ligands
    ligs = sorted(DIR_PREP.glob("*.pdbqt"))
    if not ligs:
        raise SystemExit("❌ No ligand PDBQTs found in prepared_ligands/")

    manifest = load_manifest()
    created_ts = now_iso()

    # Idempotent: skip ligands with valid existing output pose
    todo: list[Path] = []
    for lig in ligs:
        lig_id = lig.stem
        out_pose = DIR_RESULTS / f"{lig_id}_out.pdbqt"
        ok, best = pose_is_valid(out_pose)
        if ok:
            # Touch/refresh manifest row
            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["pdbqt_path"] = str(lig.resolve())
            m["vina_status"] = "DONE"
            m["vina_pose"] = str(out_pose.resolve())
            m["vina_reason"] = "OK"
            if best is not None:
                m["vina_score"] = f"{best:.2f}"
            m["config_hash"] = chash
            m["receptor_sha1"] = receptor_sha
            m["tools_vina"] = str(qvina)
            m.setdefault("created_at", created_ts)
            m["updated_at"] = now_iso()
            manifest[lig_id] = m
        else:
            todo.append(lig)

    if not todo:
        save_manifest(manifest)
        build_and_write_summaries(manifest)
        print("✅ Nothing to do — all ligands already have valid outputs.")
        print(f"Manifest: {FILE_MANIFEST}")
        print(f"Summary : {FILE_SUMMARY}")
        print(f"Leaders : {FILE_LEADER}")
        return

    print(f"🔧 QuickVina: {qvina}")
    print(f"📦 Receptor : {receptor}")
    print(f"🧊 Box      : center=({box['center_x']},{box['center_y']},{box['center_z']})  size=({box['size_x']},{box['size_y']},{box['size_z']})")
    print(f"⚙️ Params   : exhaustiveness={params['exhaustiveness']} num_modes={params['num_modes']} energy_range={params['energy_range']}")
    print(f"🧪 Jobs     : {len(todo)} ligands")

    done = failed = 0

    try:
        for i, lig in enumerate(todo, 1):
            if STOP_REQUESTED or HARD_STOP or stop_file_requested():
                print("🧾 Stop requested — finalizing after checkpoint…")
                break

            lig_id = lig.stem
            out_pose = DIR_RESULTS / f"{lig_id}_out.pdbqt"
            out_log  = DIR_RESULTS / f"{lig_id}_qvina.log"

            rc = run_qvina(qvina, receptor, lig, out_pose, out_log, box, params, quiet=True)
            ok, best = pose_is_valid(out_pose)

            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["pdbqt_path"] = str(lig.resolve())
            m["vina_pose"] = str(out_pose.resolve())
            m["vina_status"] = "DONE" if (rc == 0 and ok) else "FAILED"
            m["vina_reason"] = "OK" if (rc == 0 and ok) else ("No VINA RESULT found" if rc == 0 else f"rc={rc}")
            m["vina_score"] = f"{best:.2f}" if (rc == 0 and ok and best is not None) else ""
            m["config_hash"] = chash
            m["receptor_sha1"] = receptor_sha
            m["tools_vina"] = str(qvina)
            m.setdefault("created_at", created_ts)
            m["updated_at"] = now_iso()
            manifest[lig_id] = m

            if rc == 0 and ok:
                done += 1
            else:
                failed += 1

            # periodic checkpoints (every 50 ligands)
            if (i % 50) == 0:
                save_manifest(manifest)
                build_and_write_summaries(manifest)
                print(f"📒 Checkpoint — DONE: {done}  FAILED: {failed}")

    finally:
        save_manifest(manifest)
        build_and_write_summaries(manifest)
        print("✅ QuickVina docking complete (or safely stopped).")
        print(f"DONE: {done}  FAILED: {failed}")
        print(f"Manifest: {FILE_MANIFEST}")
        print(f"Summary : {FILE_SUMMARY}")
        print(f"Leaders : {FILE_LEADER}")

if __name__ == "__main__":
    main()
