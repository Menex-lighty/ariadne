"""Assemble the self-contained Kaggle notebook from src/perturber modules.

The notebook is a BUILD ARTIFACT — never edit it by hand; edit src/ and rebuild.
Works because of the repo's import convention: modules only use
`from perturber.x import name`, so stripping those lines leaves code that
resolves in the notebook's flat namespace.
"""
import re
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "perturber"
OUT = ROOT / "notebooks" / "kaggle_perturber.ipynb"

# Dependency order — asserted below, update when adding modules.
ORDER = ["config", "dynamics", "forces", "data", "integrators", "model",
         "fit", "evaluate", "plots", "runner"]

DRIVER = '''\
# ── Driver ────────────────────────────────────────────────────────────────
import os, glob, shutil

SMOKE = os.environ.get("PERTURBER_SMOKE") == "1"   # fast validation run

# --- What to run -----------------------------------------------------------
RUN_SWEEP  = False        # False: one demo experiment | True: detection-threshold sweep
GRID       = "core"       # "core" (48 cells, feasible) | "kaggle" (135, full science)
SHARD      = 0            # this session's shard index ...
N_SHARDS   = 1            # ... of N parallel sessions (each does cells i%N==SHARD)
# ---------------------------------------------------------------------------

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "results_nb"

if not RUN_SWEEP:
    preset = "smoke" if SMOKE else "kaggle"
    sys_cfg, fit_cfg = get_preset(preset)
    result = run_single_experiment(sys_cfg, fit_cfg, os.path.join(OUT_DIR, "experiment"))
else:
    # The sweep is CPU-bound (tiny per-step tensors -> kernel-launch overhead
    # makes a GPU slower here). Turn OFF the Kaggle accelerator for sweep runs.
    sweep_dir = os.path.join(OUT_DIR, "threshold")
    os.makedirs(sweep_dir, exist_ok=True)
    # Resume: copy any prior cell JSONs from attached input datasets so already-
    # computed cells are skipped. Attach a previous run's output as a dataset.
    for prior in glob.glob("/kaggle/input/*/threshold/*.json"):
        dst = os.path.join(sweep_dir, os.path.basename(prior))
        if not os.path.exists(dst):
            shutil.copy(prior, dst)
    run_threshold_study("smoke" if SMOKE else "kaggle", sweep_dir,
                        grid="smoke" if SMOKE else GRID,
                        shard=SHARD, n_shards=N_SHARDS)
'''


def module_body(name, idx):
    src = (SRC / f"{name}.py").read_text(encoding="utf-8")
    # Internal imports must only reference earlier modules (buildability check)
    for ref in re.findall(r"from perturber\.(\w+) import", src):
        assert ORDER.index(ref) < idx, \
            f"{name}.py imports perturber.{ref}, which comes later in ORDER"
    lines = []
    for line in src.splitlines():
        if re.match(r"\s*from perturber(\.\w+)? import", line) or \
           re.match(r"\s*import perturber", line):
            continue
        if re.match(r'if __name__ == .__main__.:', line):
            break  # drop module self-checks: __name__ IS "__main__" in a notebook
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Hidden-Perturber Discovery (Milestone 1)\n\n"
        "Recover an invisible body's mass and orbit from noisy trajectories of the "
        "visible bodies, via differentiable N-body simulation.\n\n"
        f"**BUILT from `src/perturber` at commit `{git_sha()}` — do not edit; "
        "edit the repo and rerun `scripts/build_notebook.py`.**\n\n"
        "Set `RUN_SWEEP = True` in the driver cell for the full detection-threshold "
        "study (~hours on GPU)."))
    for idx, name in enumerate(ORDER):
        nb.cells.append(nbf.v4.new_markdown_cell(f"### `src/perturber/{name}.py`"))
        nb.cells.append(nbf.v4.new_code_cell(module_body(name, idx)))
    nb.cells.append(nbf.v4.new_markdown_cell("### Run"))
    nb.cells.append(nbf.v4.new_code_cell(DRIVER))

    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, str(OUT))
    print(f"[build] {OUT}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
    sys.exit(0)
