"""
Plot model performance across neuron segments.

Reads results/summary.csv and produces:
  1. Bar chart — mean best_z_on_nerve_gained per model per segment, with SD error bars
     and run-count annotations.
  2. (Optional) Step-level z-progress curves from results/steps.csv if the file exists.

Usage:
    python plot_results.py
    python plot_results.py --summary results/summary.csv --steps results/steps.csv
    python plot_results.py --metric z_gained   # use z_gained instead of best_z_on_nerve_gained
    python plot_results.py --no-steps          # skip the step-curve plot
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── colour palette: cycles gracefully as new models are added ─────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # model column is "{base_model}_{segment}" — split on the LAST underscore cluster
    # that matches a known segment suffix, or just keep as-is if format differs.
    # We extract base_model by stripping the trailing "_<segment>" token.
    # Segment is already in position_id context; we reconstruct it from the folder name.
    # Strategy: the segment label is everything after the first underscore group of the model col
    # that isn't a digit run.  Safer: split on '_' and heuristically find the segment token.
    def split_model_segment(raw: str):
        # raw examples: "gpt-4o_original", "gpt-5_segB", "claude-sonnet_segC"
        parts = raw.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return raw, "unknown"

    df[["base_model", "segment"]] = pd.DataFrame(
        df["model"].map(split_model_segment).tolist(), index=df.index
    )
    return df


def bar_chart(df: pd.DataFrame, metric: str, out_path: str):
    """Grouped bar chart: x = segment, groups = base_model, y = mean metric."""
    segments = sorted(df["segment"].unique())
    models = sorted(df["base_model"].unique())
    n_models = len(models)
    n_segs = len(segments)

    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    # Aggregate
    grp = (
        df.groupby(["base_model", "segment"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["std"] = grp["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, n_segs * n_models * 0.9 + 2), 6))

    bar_width = 0.8 / n_models
    seg_positions = np.arange(n_segs)

    for i, model in enumerate(models):
        sub = grp[grp["base_model"] == model].set_index("segment")
        means = [sub.loc[s, "mean"] if s in sub.index else 0.0 for s in segments]
        stds  = [sub.loc[s, "std"]  if s in sub.index else 0.0 for s in segments]
        counts = [int(sub.loc[s, "count"]) if s in sub.index else 0 for s in segments]

        offsets = seg_positions + (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets, means, bar_width * 0.9,
            yerr=stds, capsize=4,
            color=color_map[model], label=model,
            error_kw={"elinewidth": 1.2, "alpha": 0.8},
        )

        # Annotate each bar with "n=<count>"
        for bar, n in zip(bars, counts):
            if n == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05 + 2,
                f"n={n}",
                ha="center", va="bottom", fontsize=7, color="#333333",
            )

    ax.set_xticks(seg_positions)
    ax.set_xticklabels(segments, fontsize=11)
    ax.set_xlabel("Neuron Segment", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=14)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def step_curves(steps_path: str, out_path: str):
    """Line plot: mean z-progress per step.

    Colour = segment (one colour per segment, shared across all models).
    Line style = model (one style per model).
    SD band shown per (model, segment) combo.
    """
    df = pd.read_csv(steps_path)

    # Reconstruct base_model / segment the same way as summary
    def split_model_segment(raw: str):
        parts = raw.rsplit("_", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (raw, "unknown")

    df[["base_model", "segment"]] = pd.DataFrame(
        df["model"].map(split_model_segment).tolist(), index=df.index
    )

    segments = sorted(df["segment"].unique())
    models   = sorted(df["base_model"].unique())

    # Colour per segment, linestyle per model
    seg_colors  = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(segments)}
    linestyles  = ["-", "--", "-.", ":"]
    model_ls    = {m: linestyles[i % len(linestyles)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for seg in segments:
        for model in models:
            sub = df[(df["base_model"] == model) & (df["segment"] == seg)]
            if sub.empty:
                continue
            per_step = sub.groupby("step")["z"].agg(["mean", "std"]).reset_index()
            color = seg_colors[seg]
            ls    = model_ls[model]
            ax.plot(
                per_step["step"], per_step["mean"],
                label=f"{seg} / {model}",
                color=color, linestyle=ls, linewidth=1.8,
            )
            ax.fill_between(
                per_step["step"],
                per_step["mean"] - per_step["std"].fillna(0),
                per_step["mean"] + per_step["std"].fillna(0),
                color=color, alpha=0.12,
            )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Z Position", fontsize=12)
    ax.set_title("Z-Progress Curves per Model × Segment", fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot VLM navigation results")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--steps",   default="results/steps.csv")
    parser.add_argument(
        "--metric",
        default="best_z_on_nerve_gained",
        help="Column from summary.csv to plot (default: best_z_on_nerve_gained)",
    )
    parser.add_argument("--no-steps", action="store_true", help="Skip step-curve plot")
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Bar chart ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.summary):
        print(f"Summary file not found: {args.summary}")
        return

    df = load_summary(args.summary)
    if args.metric not in df.columns:
        print(f"Metric '{args.metric}' not in columns: {list(df.columns)}")
        return

    bar_chart(df, args.metric, os.path.join(args.out_dir, f"bar_{args.metric}.png"))

    # ── Step curves ────────────────────────────────────────────────────────────
    if not args.no_steps:
        if os.path.exists(args.steps):
            step_curves(args.steps, os.path.join(args.out_dir, "step_curves.png"))
        else:
            print(f"Steps file not found ({args.steps}), skipping step-curve plot.")


if __name__ == "__main__":
    main()
