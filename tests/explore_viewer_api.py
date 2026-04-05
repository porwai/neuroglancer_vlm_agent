"""
Phase 1.1: Explore Neuroglancer's JS viewer API via Selenium.

Goal: Discover what window.viewer exposes so we can detect
whether a neuron segment is visible at the current position.

Runs at two positions:
  1. Default start (z=192) — segment should be visible
  2. Same x,y but z=7000 — likely empty space
"""

import time
import json
import os
from ngllib import Environment

def probe_viewer(driver, label):
    """Run a series of JS probes and print results."""
    print(f"\n{'='*60}")
    print(f"PROBING: {label}")
    print(f"{'='*60}")

    probes = {
        # Basic state
        "viewer exists": "return window.viewer !== undefined",
        "viewer type": "return typeof window.viewer",
        "viewer keys": """
            if (!window.viewer) return null;
            return Object.keys(window.viewer).slice(0, 50);
        """,

        # State object
        "viewer.state type": "return typeof viewer.state",
        "viewer.state keys": """
            if (!viewer.state) return null;
            return Object.keys(viewer.state);
        """,

        # Position from state
        "current position": """
            try {
                var s = JSON.parse(JSON.stringify(viewer.state));
                return s.position;
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Layer manager
        "layerManager exists": "return viewer.layerManager !== undefined",
        "layerManager type": "return typeof viewer.layerManager",
        "layerManager keys": """
            if (!viewer.layerManager) return null;
            return Object.keys(viewer.layerManager).slice(0, 30);
        """,

        # Managed layers
        "managedLayers": """
            try {
                if (!viewer.layerManager) return null;
                var layers = viewer.layerManager.managedLayers;
                if (!layers) return 'managedLayers is null/undefined';
                var result = [];
                for (var i = 0; i < layers.length; i++) {
                    var layer = layers[i];
                    result.push({
                        name: layer.name,
                        type: typeof layer,
                        keys: Object.keys(layer).slice(0, 20)
                    });
                }
                return result;
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Segmentation layer details
        "segmentation layer deep": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name && layers[i].name.indexOf('flywire') !== -1) {
                        var layer = layers[i];
                        var info = {
                            name: layer.name,
                            layer_keys: Object.keys(layer).slice(0, 30),
                            visible: layer.visible,
                        };
                        // Try to access the actual layer object
                        if (layer.layer) {
                            info.inner_layer_keys = Object.keys(layer.layer).slice(0, 30);
                            info.inner_layer_type = layer.layer.constructor ? layer.layer.constructor.name : typeof layer.layer;
                        }
                        return info;
                    }
                }
                return 'flywire layer not found';
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Try to access visible segments
        "visible segments": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name && layers[i].name.indexOf('flywire') !== -1) {
                        var layer = layers[i];
                        var info = {};
                        // Check various paths to segment info
                        if (layer.layer && layer.layer.displayState) {
                            info.displayState_keys = Object.keys(layer.layer.displayState).slice(0, 20);
                        }
                        if (layer.layer && layer.layer.segmentationGroupState) {
                            info.segGroupState_keys = Object.keys(layer.layer.segmentationGroupState).slice(0, 20);
                        }
                        // Try visibleSegments
                        if (layer.layer && layer.layer.displayState && layer.layer.displayState.visibleSegments) {
                            var vs = layer.layer.displayState.visibleSegments;
                            info.visibleSegments_type = typeof vs;
                            info.visibleSegments_keys = Object.keys(vs).slice(0, 20);
                            info.visibleSegments_size = vs.size;
                        }
                        // Try segmentationGroupState.visibleSegments
                        if (layer.layer && layer.layer.segmentationGroupState && layer.layer.segmentationGroupState.value) {
                            var sgv = layer.layer.segmentationGroupState.value;
                            info.sgv_keys = Object.keys(sgv).slice(0, 20);
                            if (sgv.visibleSegments) {
                                info.sgv_visibleSegments_size = sgv.visibleSegments.size;
                            }
                        }
                        return info;
                    }
                }
                return 'flywire layer not found';
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Canvas elements on page
        "canvas elements": """
            var canvases = document.querySelectorAll('canvas');
            var result = [];
            for (var i = 0; i < canvases.length; i++) {
                result.push({
                    id: canvases[i].id,
                    className: canvases[i].className,
                    width: canvases[i].width,
                    height: canvases[i].height
                });
            }
            return result;
        """,

        # Try display/panels
        "viewer.display": """
            try {
                if (!viewer.display) return 'no display';
                return Object.keys(viewer.display).slice(0, 20);
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Try to read pixel from segmentation canvas
        "canvas pixel sample": """
            try {
                var canvases = document.querySelectorAll('canvas');
                var results = [];
                for (var i = 0; i < canvases.length; i++) {
                    var c = canvases[i];
                    var gl = c.getContext('webgl2') || c.getContext('webgl');
                    if (gl) {
                        // Read a small region from center of canvas
                        var w = 10, h = 10;
                        var x = Math.floor(c.width / 2) - 5;
                        var y = Math.floor(c.height / 2) - 5;
                        var pixels = new Uint8Array(w * h * 4);
                        gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                        // Count non-zero pixels
                        var nonZero = 0;
                        for (var j = 0; j < pixels.length; j++) {
                            if (pixels[j] > 0) nonZero++;
                        }
                        results.push({
                            canvas_index: i,
                            width: c.width,
                            height: c.height,
                            center_nonzero_values: nonZero,
                            total_values: pixels.length,
                            sample_rgba: [pixels[0], pixels[1], pixels[2], pixels[3]]
                        });
                    }
                }
                return results;
            } catch(e) { return 'error: ' + e.message; }
        """,
    }

    results = {}
    for name, script in probes.items():
        try:
            result = driver.execute_script(script)
            results[name] = result
            print(f"\n--- {name} ---")
            if isinstance(result, (dict, list)):
                print(json.dumps(result, indent=2, default=str))
            else:
                print(result)
        except Exception as e:
            results[name] = f"EXCEPTION: {e}"
            print(f"\n--- {name} ---")
            print(f"EXCEPTION: {e}")

    return results


def main():
    # Use visible browser so we can see what's happening
    env = Environment(
        headless=False,
        config_path="config.json",
        verbose=True,
        reward_function=lambda s, a, ps: (1, False)
    )
    env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)

    driver = env.driver

    # Wait for initial render
    print("\nWaiting 8s for initial render...")
    time.sleep(8)

    # Save screenshot at starting position
    os.makedirs("explore_output", exist_ok=True)
    img = env.get_screenshot(save_path="explore_output/pos1_visible_z192.png")
    print(f"Screenshot saved: {img.size}")

    # Probe at starting position (z=192, segment should be visible)
    results_visible = probe_viewer(driver, "Position 1: z=192 (segment should be visible)")

    # Now move to empty space: same x,y but z=7000
    print("\n\nMoving to z=7000 (likely empty space)...")
    action_vector = [
        0, 0, 0,       # no clicks
        0, 0,           # no mouse
        0, 0, 0,        # no keys
        1,              # json_change = True
        0, 0, 6808,     # delta: move z from 192 to 7000
        0,              # no crossSectionScale change
        0, 0, 0,        # no orientation change
        0               # no projectionScale change
    ]
    env.step(action_vector)
    print("Waiting 5s for render...")
    time.sleep(5)

    img2 = env.get_screenshot(save_path="explore_output/pos2_empty_z7000.png")
    print(f"Screenshot saved: {img2.size}")

    # Probe at empty position
    results_empty = probe_viewer(driver, "Position 2: z=7000 (likely empty)")

    # Summary comparison
    print(f"\n\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")

    for key in results_visible:
        v = results_visible.get(key)
        e = results_empty.get(key)
        if v != e:
            print(f"\n--- {key} DIFFERS ---")
            print(f"  visible: {json.dumps(v, default=str)[:200]}")
            print(f"  empty:   {json.dumps(e, default=str)[:200]}")

    print("\n\nDone. Check explore_output/ for screenshots.")
    print("Press Enter to close browser...")
    input()
    env.end_session()


if __name__ == "__main__":
    main()
