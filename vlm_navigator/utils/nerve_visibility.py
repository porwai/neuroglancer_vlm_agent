"""
Low-level 2D nerve visibility detection from Neuroglancer screenshots.

This module only uses image heuristics on the left (2D) panel:
- Detect any saturated (non-grey) segmentation overlay pixels
- Remove mostly-static colored UI artifacts using a per-episode static mask
- Score dynamic colored fraction and classify visibility
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisibilityThresholds:
    """Thresholds over dynamic colored fraction in left panel."""

    not_visible_max: float = 0.0012
    visible_min: float = 0.0030


def load_image_rgb(image_path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array [H, W, 3]."""
    return np.array(Image.open(image_path).convert("RGB"))


def crop_left_panel(image_rgb: np.ndarray) -> np.ndarray:
    """Return left half (2D plane) of a Neuroglancer xy-3d screenshot."""
    return image_rgb[:, : image_rgb.shape[1] // 2, :]


def colored_mask(left_panel_rgb: np.ndarray) -> np.ndarray:
    """
    Detect any saturated (non-grey) segmentation overlay pixels.

    Rejects near-grey pixels (low chroma) that correspond to EM tissue,
    background, and UI chrome. Captures any hue used by Neuroglancer
    to color a segmentation overlay (green, cyan, orange, pink, etc.).
    """
    panel = left_panel_rgb.astype(np.int16)
    r = panel[..., 0]
    g = panel[..., 1]
    b = panel[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c          # 0 = pure grey, high = saturated
    brightness = max_c
    # Accept pixel if it has meaningful saturation and is not nearly black
    return (chroma > 35) & (brightness > 60)


# Keep old name as alias so existing callers don't break
green_mask = colored_mask


def build_static_mask(
    image_paths: Iterable[str | Path],
    freq_threshold: float = 0.9,
) -> np.ndarray:
    """
    Build per-episode static mask from frequently-colored pixels.

    Pixels colored in >= freq_threshold fraction of frames are treated as static
    UI/artifact pixels and removed from dynamic visibility scoring.
    """
    masks = []
    for path in image_paths:
        rgb = load_image_rgb(path)
        left = crop_left_panel(rgb)
        masks.append(colored_mask(left))
    if not masks:
        raise ValueError("No images provided for static-mask construction.")
    stacked = np.stack(masks, axis=0)
    return stacked.mean(axis=0) >= freq_threshold


def visibility_score(
    image_path: str | Path,
    static_mask: np.ndarray | None = None,
) -> dict:
    """
    Compute per-image visibility features from left panel.

    Returns:
      - total_colored_fraction
      - dynamic_colored_fraction
      - total_colored_pixels
      - dynamic_colored_pixels
      - left_panel_pixels

    Legacy keys (dynamic_green_fraction etc.) are included for backwards
    compatibility with existing log consumers.
    """
    rgb = load_image_rgb(image_path)
    left = crop_left_panel(rgb)
    mask = colored_mask(left)
    if static_mask is None:
        dynamic = mask
    else:
        if static_mask.shape != mask.shape:
            raise ValueError(
                f"static_mask shape {static_mask.shape} does not match image mask shape {mask.shape}"
            )
        dynamic = mask & (~static_mask)

    area = mask.size
    total_colored = int(mask.sum())
    dynamic_colored = int(dynamic.sum())
    return {
        "total_colored_fraction": float(total_colored / area),
        "dynamic_colored_fraction": float(dynamic_colored / area),
        "total_colored_pixels": total_colored,
        "dynamic_colored_pixels": dynamic_colored,
        "left_panel_pixels": int(area),
        # Legacy aliases
        "total_green_fraction": float(total_colored / area),
        "dynamic_green_fraction": float(dynamic_colored / area),
        "total_green_pixels": total_colored,
        "dynamic_green_pixels": dynamic_colored,
    }


def classify_visibility(
    dynamic_colored_fraction: float,
    thresholds: VisibilityThresholds = VisibilityThresholds(),
) -> str:
    """Classify visibility into not_visible / uncertain / visible."""
    if dynamic_colored_fraction < thresholds.not_visible_max:
        return "not_visible"
    if dynamic_colored_fraction >= thresholds.visible_min:
        return "visible"
    return "uncertain"


def step_images_in_folder(folder: str | Path) -> list[Path]:
    """List step images in folder (supports .jpg/.jpeg/.png)."""
    folder = Path(folder)
    images = list(folder.glob("step_*.jpg"))
    images += list(folder.glob("step_*.jpeg"))
    images += list(folder.glob("step_*.png"))
    return sorted(images)


def _step_number_from_name(image_path: Path) -> int | None:
    match = re.search(r"step_(\d+)\.(jpg|jpeg|png)$", image_path.name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def write_visibility_per_step_json(
    episode_folder: str | Path,
    output_name: str = "visibility_per_step.json",
    freq_threshold: float = 0.9,
    thresholds: VisibilityThresholds = VisibilityThresholds(),
) -> Path:
    """
    Compute 2D-plane nerve visibility for every step image and write JSON.

    Output shape:
    {
      "episode_folder": "...",
      "config": {...},
      "steps": [
        {
          "step": 1,
          "image": "step_001.jpg",
          "label": "visible|uncertain|not_visible",
          "dynamic_colored_fraction": ...,
          "total_colored_fraction": ...,
          ...
        }
      ]
    }
    """
    folder = Path(episode_folder)
    images = step_images_in_folder(folder)
    if not images:
        raise ValueError(f"No step images found in: {folder}")

    static_mask = build_static_mask(images, freq_threshold=freq_threshold)
    steps = []
    for image_path in images:
        step_num = _step_number_from_name(image_path)
        score = visibility_score(image_path, static_mask=static_mask)
        label = classify_visibility(
            score["dynamic_colored_fraction"],
            thresholds=thresholds,
        )
        steps.append(
            {
                "step": step_num,
                "image": image_path.name,
                "label": label,
                **score,
            }
        )

    payload = {
        "episode_folder": str(folder),
        "config": {
            "freq_threshold": freq_threshold,
            "not_visible_max": thresholds.not_visible_max,
            "visible_min": thresholds.visible_min,
        },
        "steps": steps,
    }
    out_path = folder / output_name
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
