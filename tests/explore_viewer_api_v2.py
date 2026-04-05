"""
Phase 1.1b: Deeper probe of Neuroglancer's segmentation layer internals.

Focus: Can we find segment bounding box, mesh extent, or chunk-level
visibility info that tells us if the neuron exists at the current Z?

Tests at 3 positions:
  1. z=192   — neuron visible in 2D cross-section
  2. z=3870  — near soma, neuron should be visible
  3. z=7000  — beyond data range, empty
"""

import time
import json
import os
from ngllib import Environment


def deep_probe(driver, label):
    """Targeted probes for segment spatial info."""
    print(f"\n{'='*60}")
    print(f"PROBING: {label}")
    print(f"{'='*60}")

    probes = {
        # What position are we at?
        "position": """
            return JSON.parse(JSON.stringify(viewer.state)).position;
        """,

        # renderLayers on segmentation layer
        "renderLayers": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var rl = layers[i].layer_.renderLayers;
                        if (!rl) return 'renderLayers is null';
                        var result = [];
                        for (var j = 0; j < rl.length; j++) {
                            var r = rl[j];
                            result.push({
                                index: j,
                                type: r.constructor ? r.constructor.name : typeof r,
                                keys: Object.keys(r).slice(0, 30)
                            });
                        }
                        return result;
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # dataSources on segmentation layer
        "dataSources": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var ds = layers[i].layer_.dataSources;
                        if (!ds) return 'dataSources is null';
                        var result = [];
                        for (var j = 0; j < ds.length; j++) {
                            var d = ds[j];
                            result.push({
                                index: j,
                                keys: Object.keys(d).slice(0, 30),
                                loadState_type: d.loadState ? (d.loadState.constructor ? d.loadState.constructor.name : typeof d.loadState) : 'null',
                                loadState_keys: d.loadState ? Object.keys(d.loadState).slice(0, 30) : null
                            });
                        }
                        return result;
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Try to get segment bounding box or spatial extent from loadState
        "loadState deep": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var ds = layers[i].layer_.dataSources;
                        for (var j = 0; j < ds.length; j++) {
                            var ls = ds[j].loadState;
                            if (!ls) continue;
                            var info = {ds_index: j, keys: Object.keys(ls).slice(0, 30)};
                            // modelTransform might have bounds
                            if (ls.modelTransform) {
                                info.modelTransform_keys = Object.keys(ls.modelTransform);
                            }
                            // subsourceEntries
                            if (ls.subsourceEntries) {
                                var subs = [];
                                ls.subsourceEntries.forEach(function(v, k) {
                                    subs.push({key: k, type: v.constructor ? v.constructor.name : typeof v, keys: Object.keys(v).slice(0, 20)});
                                });
                                info.subsourceEntries = subs;
                            }
                            return info;
                        }
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Try to get the volume bounding box
        "volume bounds": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var ds = layers[i].layer_.dataSources;
                        for (var j = 0; j < ds.length; j++) {
                            var ls = ds[j].loadState;
                            if (!ls) continue;
                            if (ls.modelTransform) {
                                var mt = ls.modelTransform;
                                return {
                                    unpaddedRank: mt.unpaddedRank,
                                    rank: mt.rank,
                                    sourceRank: mt.sourceRank,
                                    inputSpace: mt.inputSpace ? {
                                        names: mt.inputSpace.names,
                                        units: mt.inputSpace.units,
                                        scales: Array.from(mt.inputSpace.scales || []),
                                        boundingBoxes: mt.inputSpace.boundingBoxes ? mt.inputSpace.boundingBoxes.map(function(bb) {
                                            return {lowerBounds: Array.from(bb.lowerBounds), upperBounds: Array.from(bb.upperBounds)};
                                        }) : null,
                                        bounds: mt.inputSpace.bounds ? {
                                            lowerBounds: Array.from(mt.inputSpace.bounds.lowerBounds),
                                            upperBounds: Array.from(mt.inputSpace.bounds.upperBounds)
                                        } : null
                                    } : null
                                };
                            }
                        }
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Segment selection state — does it know which segments are loaded?
        "segmentSelectionState": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var ds = layers[i].layer_.displayState;
                        if (!ds) return 'no displayState';
                        var info = {};
                        if (ds.segmentSelectionState) {
                            var ss = ds.segmentSelectionState;
                            info.segSelState_keys = Object.keys(ss).slice(0, 20);
                            info.hasSelectedSegment = ss.hasSelectedSegment;
                            info.rawValue = ss.rawValue ? Array.from(ss.rawValue) : null;
                            info.value = ss.value !== undefined ? String(ss.value) : null;
                        }
                        info.hasVolume_value = ds.hasVolume ? ds.hasVolume.value : null;
                        return info;
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # renderLayers[0] deep dive — chunk visibility
        "renderLayer0 deep": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var rl = layers[i].layer_.renderLayers;
                        if (!rl || rl.length === 0) return 'no renderLayers';
                        var r = rl[0];
                        var info = {
                            type: r.constructor ? r.constructor.name : typeof r,
                            keys: Object.keys(r).slice(0, 40)
                        };
                        // Check for source, visibleChunks, etc.
                        if (r.source) {
                            info.source_type = r.source.constructor ? r.source.constructor.name : typeof r.source;
                            info.source_keys = Object.keys(r.source).slice(0, 30);
                            // chunks map?
                            if (r.source.chunks) {
                                var chunkCount = 0;
                                r.source.chunks.forEach(function() { chunkCount++; });
                                info.source_chunk_count = chunkCount;
                            }
                        }
                        return info;
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Dig into ALL renderLayers sources for chunk counts
        "all renderLayers chunks": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('flywire') !== -1) {
                        var rl = layers[i].layer_.renderLayers;
                        if (!rl) return 'no renderLayers';
                        var results = [];
                        for (var j = 0; j < rl.length; j++) {
                            var r = rl[j];
                            var entry = {
                                index: j,
                                type: r.constructor ? r.constructor.name : typeof r
                            };
                            if (r.source) {
                                entry.source_type = r.source.constructor ? r.source.constructor.name : typeof r.source;
                                if (r.source.chunks) {
                                    var count = 0;
                                    r.source.chunks.forEach(function() { count++; });
                                    entry.chunk_count = count;
                                }
                            }
                            // Check for visibleChunks or similar
                            if (r.visibleChunks !== undefined) {
                                entry.visibleChunks = r.visibleChunks;
                            }
                            results.push(entry);
                        }
                        return results;
                    }
                }
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Try layerSelectedValues — does it report what's under the cursor?
        "layerSelectedValues": """
            try {
                var lsv = viewer.layerSelectedValues;
                if (!lsv) return 'null';
                return {
                    keys: Object.keys(lsv).slice(0, 20),
                    type: lsv.constructor ? lsv.constructor.name : typeof lsv
                };
            } catch(e) { return 'error: ' + e.message; }
        """,

        # Check the image layer too — does it have data bounds?
        "image layer bounds": """
            try {
                var layers = viewer.layerManager.managedLayers;
                for (var i = 0; i < layers.length; i++) {
                    if (layers[i].name.indexOf('Maryland') !== -1) {
                        var ds = layers[i].layer_.dataSources;
                        for (var j = 0; j < ds.length; j++) {
                            var ls = ds[j].loadState;
                            if (!ls || !ls.modelTransform) continue;
                            var mt = ls.modelTransform;
                            if (mt.inputSpace) {
                                return {
                                    names: mt.inputSpace.names,
                                    scales: Array.from(mt.inputSpace.scales || []),
                                    boundingBoxes: mt.inputSpace.boundingBoxes ? mt.inputSpace.boundingBoxes.map(function(bb) {
                                        return {lowerBounds: Array.from(bb.lowerBounds), upperBounds: Array.from(bb.upperBounds)};
                                    }) : null
                                };
                            }
                        }
                    }
                }
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


def move_to_z(env, target_z, current_z):
    """Move to a target Z position."""
    delta_z = target_z - current_z
    action = [
        0, 0, 0,
        0, 0,
        0, 0, 0,
        1,
        0, 0, delta_z,
        0,
        0, 0, 0,
        0
    ]
    env.step(action)
    time.sleep(5)


def main():
    env = Environment(
        headless=False,
        config_path="config.json",
        verbose=False,
        reward_function=lambda s, a, ps: (1, False)
    )
    env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)
    driver = env.driver

    os.makedirs("explore_output", exist_ok=True)

    # Position 1: z=192 (default, segment visible)
    print("Waiting 10s for initial render...")
    time.sleep(10)
    env.get_screenshot(save_path="explore_output/v2_z192.png")
    r1 = deep_probe(driver, "z=192 — segment visible")

    # Position 2: z=3870 (near soma)
    print("\nMoving to z=3870...")
    move_to_z(env, 3870, 192)
    env.get_screenshot(save_path="explore_output/v2_z3870.png")
    r2 = deep_probe(driver, "z=3870 — near soma")

    # Position 3: z=7000 (beyond data)
    print("\nMoving to z=7000...")
    move_to_z(env, 7000, 3870)
    env.get_screenshot(save_path="explore_output/v2_z7000.png")
    r3 = deep_probe(driver, "z=7000 — beyond data range")

    # Comparison
    print(f"\n\n{'='*60}")
    print("KEY DIFFERENCES")
    print(f"{'='*60}")
    for key in r1:
        vals = [r1.get(key), r2.get(key), r3.get(key)]
        if not (vals[0] == vals[1] == vals[2]):
            print(f"\n--- {key} ---")
            print(f"  z=192:  {json.dumps(vals[0], default=str)[:200]}")
            print(f"  z=3870: {json.dumps(vals[1], default=str)[:200]}")
            print(f"  z=7000: {json.dumps(vals[2], default=str)[:200]}")

    print("\n\nDone. Press Enter to close browser...")
    input()
    env.end_session()


if __name__ == "__main__":
    main()
