# About MolDockPipe

MolDockPipe is a Linux-oriented toolkit of standalone scripts for small-molecule
virtual screening. Each numbered module covers cheminformatics preprocessing
(RDKit-based descriptor calculation, ADMET filtering, and conformer generation),
Meeko ligand preparation, or AutoDock Vina docking. Outputs are still recorded
in a manifest so runs can be resumed, audited, or repeated deterministically.

## Highlights
- **Python 3.11 first**: the modules assume Python 3.11's standard-library
  behavior and typing features.
- **Modular stages**: each numbered `Module *.py` file can be executed on its
  own or chained manually in sequence for a full screening cycle.
- **No bundled orchestrator**: the previous `run_pipeline.py` helper has been
  removed, so automation is left to external schedulers or manual invocation.
- **Reproducibility baked in**: outputs, failure reasons, and tool versions are
  tracked in `state/manifest.csv`, making downstream triage or troubleshooting
  straightforward.

## Folder-at-a-Glance
- `input/` — CSV of SMILES and metadata for candidate ligands.
- `3D_Structures/` — RDKit-generated 3D conformers ready for refinement.
- `prepared_ligands/` — Meeko-converted PDBQT files for docking.
- `receptors/` — Target structures paired with Vina configuration files.
- `results/` — Docked poses, Vina logs, and leaderboards.
- `state/` — Manifest, pass/fail lists, and run metadata.

## Getting Started
1. Install Python 3.11 and project dependencies (`rdkit-pypi`, `meeko`,
   `pyyaml` when using YAML configs, plus the AutoDock Vina binary for your
   hardware).
2. Populate `input/input.csv` with ligand identifiers, SMILES strings, and
   optional per-ligand overrides in JSON form.
3. Execute `Module 1.py` through `Module 4.py` manually in sequence (or script
   your own orchestration, if desired).
4. Use `WARNING_PURGE_PIPELINE.py` whenever you need to reset generated
   artifacts while keeping directory scaffolding in place.

MolDockPipe targets computational chemists who want a reproducible, scriptable
workflow without locking themselves into a GUI-heavy suite.
