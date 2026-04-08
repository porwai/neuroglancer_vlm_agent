"""
Generate per-step 2D nerve visibility JSON for a manual-test episode folder.

Example:
    python generate_visibility_json.py --folder results/manual_test/gpt-5_pos1
"""

import argparse

from vlm_navigator.utils.nerve_visibility import (
    VisibilityThresholds,
    write_visibility_per_step_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write visibility_per_step.json from step images")
    parser.add_argument("--folder", required=True, help="Episode folder containing step_*.jpg/png")
    parser.add_argument("--freq-threshold", type=float, default=0.9, help="Static-mask green frequency threshold")
    parser.add_argument("--not-visible-max", type=float, default=0.0012, help="Max dynamic fraction for not_visible")
    parser.add_argument("--visible-min", type=float, default=0.0030, help="Min dynamic fraction for visible")
    parser.add_argument("--output-name", default="visibility_per_step.json", help="Output JSON filename")
    args = parser.parse_args()

    thresholds = VisibilityThresholds(
        not_visible_max=args.not_visible_max,
        visible_min=args.visible_min,
    )
    out_path = write_visibility_per_step_json(
        episode_folder=args.folder,
        output_name=args.output_name,
        freq_threshold=args.freq_threshold,
        thresholds=thresholds,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
