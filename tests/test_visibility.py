"""
Phase 1.3: Test segment visibility detection.

Approach: Toggle the segmentation layer visibility, take screenshots
with and without it, and diff. Any pixel difference in the 2D
cross-section panel means the neuron is rendered at this Z.

Also tests simple "is the 2D panel non-black" as a data-presence check.

Tests at 4 positions:
  1. z=192   — neuron clearly visible
  2. z=3870  — near soma, neuron barely visible
  3. z=5500  — EM data exists but neuron may not be present
  4. z=7000  — beyond data range
"""

import time
import json
import os
import numpy as np
from PIL import Image
from ngllib import Environment


def get_left_panel(image):
    """Crop the 2D cross-section panel (left half of xy-3d layout).
    Exclude the top toolbar area (~40px) and bottom scale bar (~30px)."""
    w, h = image.size
    # Left panel is roughly the left half, excluding UI chrome
    top_margin = 50
    bottom_margin = 35
    return image.crop((0, top_margin, w // 2, h - bottom_margin))


def check_panel_has_content(panel_image):
    """Check if the 2D panel has any non-black content.
    Returns the fraction of pixels that are non-black."""
    arr = np.array(panel_image)
    # A pixel is "non-black" if any channel > threshold
    # Use threshold of 10 to ignore near-black noise
    non_black = np.any(arr > 10, axis=2)
    fraction = np.mean(non_black)
    return fraction


def check_segment_visible_by_diff(driver, env, label):
    """Toggle segmentation layer, take before/after screenshots, diff.
    Returns (visible: bool, diff_fraction: float, panel_content_fraction: float)."""

    # Screenshot WITH segmentation
    time.sleep(0.5)
    img_with = env.get_screenshot()
    panel_with = get_left_panel(img_with)

    # Toggle segmentation layer OFF
    driver.execute_script("""
        var layers = viewer.layerManager.managedLayers;
        for (var i = 0; i < layers.length; i++) {
            if (layers[i].name.indexOf('flywire') !== -1) {
                layers[i].visible = false;
                break;
            }
        }
    """)
    time.sleep(1.0)  # wait for re-render

    # Screenshot WITHOUT segmentation
    img_without = env.get_screenshot()
    panel_without = get_left_panel(img_without)

    # Toggle segmentation layer back ON
    driver.execute_script("""
        var layers = viewer.layerManager.managedLayers;
        for (var i = 0; i < layers.length; i++) {
            if (layers[i].name.indexOf('flywire') !== -1) {
                layers[i].visible = true;
                break;
            }
        }
    """)
    time.sleep(0.5)

    # Compute diff
    arr_with = np.array(panel_with).astype(float)
    arr_without = np.array(panel_without).astype(float)
    diff = np.abs(arr_with - arr_without)
    # A pixel has meaningful diff if any channel differs by > 5
    diff_pixels = np.any(diff > 5, axis=2)
    diff_fraction = np.mean(diff_pixels)

    # Also check raw content
    content_fraction = check_panel_has_content(panel_with)

    # Save debug images
    os.makedirs("explore_output/visibility", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("=", "")
    panel_with.save(f"explore_output/visibility/{safe_label}_with_seg.png")
    panel_without.save(f"explore_output/visibility/{safe_label}_without_seg.png")
    # Save diff as image
    diff_img = Image.fromarray((diff_pixels * 255).astype(np.uint8))
    diff_img.save(f"explore_output/visibility/{safe_label}_diff.png")

    print(f"\n  {label}:")
    print(f"    Panel content (non-black): {content_fraction:.4f} ({content_fraction*100:.1f}%)")
    print(f"    Segmentation diff:         {diff_fraction:.6f} ({diff_fraction*100:.3f}%)")
    print(f"    Segment visible:           {diff_fraction > 0.001}")

    return diff_fraction > 0.001, diff_fraction, content_fraction


def move_to_z(env, target_z, current_z):
    """Move to a target Z position."""
    delta_z = target_z - current_z
    action = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, delta_z, 0, 0, 0, 0, 0]
    env.step(action)
    time.sleep(3)


def main():
    env = Environment(
        headless=False,
        config_path="config.json",
        verbose=False,
        reward_function=lambda s, a, ps: (1, False)
    )
    env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)
    driver = env.driver

    print("Waiting 10s for initial render...")
    time.sleep(10)

    positions = [
        (192, "z=192 neuron visible"),
        (3870, "z=3870 near soma"),
        (5500, "z=5500 maybe no neuron"),
        (7000, "z=7000 beyond data"),
    ]

    results = []
    current_z = 192

    for target_z, label in positions:
        if target_z != current_z:
            print(f"\nMoving to z={target_z}...")
            move_to_z(env, target_z, current_z)
            current_z = target_z

        visible, diff_frac, content_frac = check_segment_visible_by_diff(
            driver, env, label
        )
        results.append((label, visible, diff_frac, content_frac))

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Position':<30} {'Content%':>10} {'SegDiff%':>10} {'Visible':>8}")
    print("-" * 60)
    for label, visible, diff_frac, content_frac in results:
        print(f"{label:<30} {content_frac*100:>9.1f}% {diff_frac*100:>9.3f}% {'YES' if visible else 'NO':>8}")

    print("\n\nCheck explore_output/visibility/ for debug images.")
    print("Press Enter to close browser...")
    input()
    env.end_session()


if __name__ == "__main__":
    main()
