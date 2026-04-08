"""
Low-level 2D nerve visibility detection from Neuroglancer screenshots.

This module only uses image heuristics on the left (2D) panel:
- Detect green segmentation-like pixels
- Remove mostly-static green UI artifacts using a per-episode static mask
- Score dynamic green fraction and classify visibility
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
    """Thresholds over dynamic green fraction in left panel."""

    not_visible_max: float = 0.0012
    visible_min: float = 0.0030


def load_image_rgb(image_path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array [H, W, 3]."""
    return np.array(Image.open(image_path).convert("RGB"))


def crop_left_panel(image_rgb: np.ndarray) -> np.ndarray:
    """Return left half (2D plane) of a Neuroglancer xy-3d screenshot."""
    return image_rgb[:, : image_rgb.shape[1] // 2, :]


def green_mask(left_panel_rgb: np.ndarray) -> np.ndarray:
    """
    Detect green-ish segmentation overlay pixels.

    Uses an RGB dominance + saturation-style heuristic to capture overlay pixels
    while rejecting most grayscale tissue/background pixels.
    """
    panel = left_panel_rgb.astype(np.int16)
    r = panel[..., 0]
    g = panel[..., 1]
    b = panel[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat_like = max_c - min_c
    return (g > r + 20) & (g > b + 20) & (g > 90) & (sat_like > 35)


def build_static_mask(
    image_paths: Iterable[str | Path],
    freq_threshold: float = 0.9,
) -> np.ndarray:
    """
    Build per-episode static mask from frequently-green pixels.

    Pixels green in >= freq_threshold fraction of frames are treated as static
    UI/artifact pixels and removed from dynamic visibility scoring.
    """
    masks = []
    for path in image_paths:
        rgb = load_image_rgb(path)
        left = crop_left_panel(rgb)
        masks.append(green_mask(left))
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
      - total_green_fraction
      - dynamic_green_fraction
      - total_green_pixels
      - dynamic_green_pixels
      - left_panel_pixels
    """
    rgb = load_image_rgb(image_path)
    left = crop_left_panel(rgb)
    mask = green_mask(left)
    if static_mask is None:
        dynamic = mask
    else:
        if static_mask.shape != mask.shape:
            raise ValueError(
                f"static_mask shape {static_mask.shape} does not match image mask shape {mask.shape}"
            )
        dynamic = mask & (~static_mask)

    area = mask.size
    total_green = int(mask.sum())
    dynamic_green = int(dynamic.sum())
    return {
        "total_green_fraction": float(total_green / area),
        "dynamic_green_fraction": float(dynamic_green / area),
        "total_green_pixels": total_green,
        "dynamic_green_pixels": dynamic_green,
        "left_panel_pixels": int(area),
    }


def classify_visibility(
    dynamic_green_fraction: float,
    thresholds: VisibilityThresholds = VisibilityThresholds(),
) -> str:
    """Classify visibility into not_visible / uncertain / visible."""
    if dynamic_green_fraction < thresholds.not_visible_max:
        return "not_visible"
    if dynamic_green_fraction >= thresholds.visible_min:
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
          "dynamic_green_fraction": ...,
          "total_green_fraction": ...,
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
            score["dynamic_green_fraction"],
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
