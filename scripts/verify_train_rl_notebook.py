"""Local smoke test for kaggle/train_rl_notebook.ipynb.

Extracts the notebook's actual code cells (not a hand-retyped copy — read
straight from the .ipynb JSON, so this can't silently drift from what's
really in the notebook), patches a couple of scale constants down to
something that finishes in seconds on a laptop CPU, and executes them in
one namespace exactly like a Jupyter kernel would run the cells in order.

This cannot exercise Kaggle's GPU, multi-day training runs, or the
`/kaggle/input/...` dataset paths — nobody has real Kaggle credentials to
hand this environment — but it does prove every import resolves, every API
call matches the real function signatures, and a full
vecenv-reset-with-a-freshly-generated-netlist -> PPO forward/backward pass
completes without error.
"""

import json
import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
NOTEBOOK_PATH = os.path.join(REPO_ROOT, "kaggle", "train_rl_notebook.ipynb")

PATCHES = {
    "NUM_ENVS = 4  # Kaggle notebooks typically get 4 CPU cores": "NUM_ENVS = 2  # reduced for local smoke test",
    "save_freq=max(1, 10_000 // NUM_ENVS),": "save_freq=max(1, 256 // NUM_ENVS),",
    "TOTAL_TIMESTEPS = 100_000": "TOTAL_TIMESTEPS = 256  # reduced for local smoke test",
}


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    os.environ["PCB_GEN_REPO_ROOT"] = REPO_ROOT

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"Extracted {len(code_cells)} code cells from {NOTEBOOK_PATH}")

    namespace = {"__name__": "__main__"}
    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        # strip Jupyter shell-escape lines (e.g. "!pip install ...") — not
        # valid Python syntax outside a notebook kernel; already installed
        # locally anyway.
        source = "\n".join(line for line in source.splitlines() if not line.strip().startswith("!"))
        if not source.strip():
            print(f"--- cell {i}: nothing left after stripping shell lines, skipping ---")
            continue
        for old, new in PATCHES.items():
            if old in source:
                source = source.replace(old, new)
                print(f"--- cell {i}: patched '{old.splitlines()[0]}' -> '{new}' ---")
        print(f"--- executing cell {i} ({len(source)} chars) ---")
        exec(compile(source, f"<notebook cell {i}>", "exec"), namespace)

    print("\nAll notebook code cells executed successfully.")
    print(f"Final model type: {type(namespace['model'])}")


if __name__ == "__main__":
    sys.exit(main())
