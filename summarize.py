"""
Aggregate all log.json files under results/manual_test/ into a CSV.

Produces two files in results/:
  summary.csv      — one row per run (model, position, trial, z_gained, steps, etc.)
  steps.csv        — one row per step across all runs (for z-trajectory plots)

Usage:
    python summarize.py
    python summarize.py --results-dir results/manual_test/my-run-id
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path


def parse_folder_name(folder_name: str) -> dict:
    """Best-effort parse of model/position/trial from folder name."""
    m = re.match(r"^(.+?)_pos(\d+)_trial(\d+)$", folder_name)
    if m:
        return {"model": m.group(1), "position_id": int(m.group(2)), "trial": int(m.group(3))}
    m = re.match(r"^(.+?)_pos(\d+)$", folder_name)
    if m:
        return {"model": m.group(1), "position_id": int(m.group(2)), "trial": 1}
    return {"model": folder_name, "position_id": None, "trial": 1}


def collect_runs(results_dir: Path) -> list[dict]:
    runs = []
    for log_path in sorted(results_dir.rglob("log.json")):
        folder = log_path.parent
        with open(log_path) as f:
            steps = json.load(f)
        if not steps:
            continue

        meta = parse_folder_name(folder.name)
        start_z = steps[0]["position_before"][2]
        final_z = steps[-1]["position_after"][2]
        early_stop = any(s.get("early_stop") for s in steps)
        agent_stop = steps[-1].get("action") is not None and bool(
            (steps[-1].get("action") or {}).get("done", False)
        )

        # nerve visibility counts and best z while nerve was on-screen
        nerve_counts = {"visible": 0, "uncertain": 0, "not_visible": 0}
        best_z_on_nerve = start_z
        for s in steps:
            label = s.get("nerve_visible")
            if label in nerve_counts:
                nerve_counts[label] += 1
            if label in ("visible", "uncertain"):
                z = s["position_after"][2]
                if z > best_z_on_nerve:
                    best_z_on_nerve = z

        runs.append({
            "folder": str(folder.relative_to(results_dir.parent)),
            "model": meta["model"],
            "position_id": meta["position_id"],
            "trial": meta["trial"],
            "steps_taken": len(steps),
            "start_z": start_z,
            "final_z": final_z,
            "z_gained": final_z - start_z,
            "best_z_on_nerve": best_z_on_nerve,
            "best_z_on_nerve_gained": best_z_on_nerve - start_z,
            "early_stop_nerve": early_stop,
            "agent_stop": agent_stop,
            "steps_visible": nerve_counts["visible"],
            "steps_uncertain": nerve_counts["uncertain"],
            "steps_not_visible": nerve_counts["not_visible"],
        })
    return runs


def collect_steps(results_dir: Path, runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        log_path = results_dir.parent / run["folder"] / "log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            steps = json.load(f)
        for s in steps:
            rows.append({
                "folder": run["folder"],
                "model": run["model"],
                "position_id": run["position_id"],
                "trial": run["trial"],
                "step": s["step"],
                "z": s["position_after"][2],
                "z_delta": s.get("z_delta", 0),
                "nerve_visible": s.get("nerve_visible", ""),
                "early_stop": s.get("early_stop", False),
            })
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        print(f"  No data for {path.name}, skipping.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results/manual_test",
        help="Root directory to scan for log.json files (default: results/manual_test)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        raise SystemExit(1)

    out_dir = Path("results")
    runs = collect_runs(results_dir)
    steps = collect_steps(results_dir, runs)

    print(f"Found {len(runs)} runs, {len(steps)} total steps.")
    write_csv(out_dir / "summary.csv", runs)
    write_csv(out_dir / "steps.csv", steps)
