#!/usr/bin/env python3
"""Reproduce the paper figures used in paper/PAPP_arxiv.tex.

Runs the repository's existing generation/validation scripts and then verifies
that every file referenced by \\includegraphics{...} in the TeX exists on disk.

Usage (from repo root):
  python reproduce_paper.py

Prereqs:
  pip install -r requirements.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TEX_FILE = REPO_ROOT / "paper" / "PAPP_arxiv.tex"


def run_step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    print("$", " ".join(args))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(args, cwd=REPO_ROOT, check=True, env=env)


def iter_includegraphics_paths(tex_text: str) -> list[str]:
    # Matches: \includegraphics{path} and \includegraphics[...]{path}
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    return sorted({m.group(1).strip() for m in re.finditer(pattern, tex_text) if m.group(1).strip()})


def main() -> int:
    if not TEX_FILE.exists():
        print(f"ERROR: TeX file not found: {TEX_FILE}", file=sys.stderr)
        return 2

    python = sys.executable

    # Keep this intentionally minimal: run the scripts that are already used to
    # generate/validate the figures referenced by the paper.
    try:
        run_step("Quickstart sanity run", [python, "-X", "utf8", "examples/quickstart.py"])
        run_step("Generate paper figures", [python, "-X", "utf8", "code_src/generate_paper_figures.py"])
        run_step("Component count validation", [python, "-X", "utf8", "code_src/component_count_validation.py"])
        run_step("Optimal phase boundaries", [python, "-X", "utf8", "code_src/calculate_optimal_phase_boundaries.py"])
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Step failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1

    tex_text = TEX_FILE.read_text(encoding="utf-8", errors="replace")
    inc_paths = iter_includegraphics_paths(tex_text)

    paper_dir = TEX_FILE.parent
    missing: list[str] = []

    for rel in inc_paths:
        target = (paper_dir / rel).resolve()
        if not target.exists():
            missing.append(rel)

    if missing:
        print("\nERROR: Missing figure files referenced by \\includegraphics:")
        for p in missing:
            print("  -", p)
        return 3

    print("\nOK: All \\includegraphics targets exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
