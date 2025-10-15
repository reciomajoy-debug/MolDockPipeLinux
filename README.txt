Automated Virtual Screening Orchestrator
=======================================

Files
-----
- run_pipeline.py — main orchestrator
- utils_text.py   — emoji/Unicode sanitizer

How to use
----------
1) Place run_pipeline.py and utils_text.py in the SAME FOLDER as your module scripts:
     Module 1.py, Module 2.py, Module 3.py, ...
2) From that folder, run one of:
     python run_pipeline.py --fresh      # full fresh run
     python run_pipeline.py --resume     # resume from last success
     python run_pipeline.py --list       # show detected modules
     python run_pipeline.py --only 3,4   # run only stages 3 and 4
     python run_pipeline.py --skip 1     # skip stage 1

Notes
-----
- Logs are saved to _pipeline_logs/raw and _pipeline_logs/clean (emoji-free).
- Checkpoints are saved to _pipeline_checkpoints/ (one .done file per finished stage).
- The orchestrator forces UTF-8 for child processes to reduce Windows encoding issues.
