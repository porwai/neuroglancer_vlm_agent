"""
Phase 1 round-trip test.

Tests:
  1. parse_vlm_response — various input formats
  2. vlm_json_to_action_vector — mode B and mode A
  3. starting_positions.json — valid data
  4. Live env round-trip: JSON → vector → env.step() → verify position delta
"""

import sys
import json
import time

# ---- Unit tests (no env needed) ----

def test_parse_vlm_response():
    from vlm_navigator.utils.action_utils import parse_vlm_response, FALLBACK_ACTION

    print("=== test_parse_vlm_response ===")

    # Pure JSON
    r = parse_vlm_response('{"delta_x": 10, "delta_y": 0, "delta_z": 50}')
    assert r == {"delta_x": 10, "delta_y": 0, "delta_z": 50}, f"Failed pure JSON: {r}"
    print("  pure JSON: OK")

    # Markdown code block
    r = parse_vlm_response('Here is my action:\n```json\n{"delta_x": 0, "delta_y": -5, "delta_z": 100}\n```\nLet me explain...')
    assert r == {"delta_x": 0, "delta_y": -5, "delta_z": 100}, f"Failed markdown: {r}"
    print("  markdown block: OK")

    # Code block without json tag
    r = parse_vlm_response('```\n{"delta_z": 25}\n```')
    assert r == {"delta_z": 25}, f"Failed bare code block: {r}"
    print("  bare code block: OK")

    # JSON embedded in text
    r = parse_vlm_response('I think we should move up. {"delta_x": 0, "delta_y": 0, "delta_z": 30} That should work.')
    assert r == {"delta_x": 0, "delta_y": 0, "delta_z": 30}, f"Failed embedded: {r}"
    print("  embedded JSON: OK")

    # Unparseable → fallback
    r = parse_vlm_response("I don't know what to do, let's just go up")
    assert r == FALLBACK_ACTION, f"Failed fallback: {r}"
    print("  fallback on garbage: OK")

    # Empty string
    r = parse_vlm_response("")
    assert r == FALLBACK_ACTION, f"Failed empty: {r}"
    print("  fallback on empty: OK")

    print("  ALL PASSED\n")


def test_vlm_json_to_action_vector():
    from vlm_navigator.utils.action_utils import vlm_json_to_action_vector

    print("=== test_vlm_json_to_action_vector ===")

    # Mode B (position_only)
    vec = vlm_json_to_action_vector({"delta_x": 10, "delta_y": -5, "delta_z": 50})
    assert len(vec) == 17, f"Wrong length: {len(vec)}"
    assert vec[8] == 1, "json_change should be 1"
    assert vec[9] == 10, f"delta_x wrong: {vec[9]}"
    assert vec[10] == -5, f"delta_y wrong: {vec[10]}"
    assert vec[11] == 50, f"delta_z wrong: {vec[11]}"
    assert vec[12] == 0, "dcross should be 0 in mode B"
    assert vec[13:16] == [0, 0, 0], "orientation should be 0 in mode B"
    assert vec[16] == 0, "dproj should be 0 in mode B"
    print("  mode B: OK")

    # Mode A (full)
    vec = vlm_json_to_action_vector(
        {"delta_x": 1, "delta_y": 2, "delta_z": 3,
         "delta_e1": 0.1, "delta_e2": 0.2, "delta_e3": 0.3,
         "delta_cross": 0.5, "delta_proj": 100},
        mode="full"
    )
    assert vec[9:12] == [1, 2, 3], f"Position wrong: {vec[9:12]}"
    assert vec[12] == 0.5, f"dcross wrong: {vec[12]}"
    assert vec[13:16] == [0.1, 0.2, 0.3], f"Orientation wrong: {vec[13:16]}"
    assert vec[16] == 100, f"dproj wrong: {vec[16]}"
    print("  mode A: OK")

    # Missing keys default to 0
    vec = vlm_json_to_action_vector({"delta_z": 50})
    assert vec[9] == 0 and vec[10] == 0 and vec[11] == 50
    print("  missing keys default to 0: OK")

    print("  ALL PASSED\n")


def test_starting_positions():
    import os
    print("=== test_starting_positions ===")

    path = os.path.join(os.path.dirname(__file__), "vlm_navigator", "config", "starting_positions.json")
    with open(path) as f:
        positions = json.load(f)

    assert len(positions) == 16, f"Expected 16 positions, got {len(positions)}"
    for p in positions:
        assert all(k in p for k in ("id", "name", "x", "y", "z", "category")), f"Missing keys in {p}"
    print(f"  {len(positions)} positions loaded, all have required keys: OK")
    print("  ALL PASSED\n")


# ---- Live env test (requires Chrome + Neuroglancer) ----

def test_env_roundtrip():
    """Live round-trip: send action vectors, verify position changes.

    Uses env's built-in change_JSON_state_url() to set starting position
    instead of a separate url_builder.
    """
    from ngllib import Environment
    from vlm_navigator.utils.action_utils import parse_vlm_response, vlm_json_to_action_vector

    print("=== test_env_roundtrip (live) ===")

    # Load starting positions
    with open("vlm_navigator/config/starting_positions.json") as f:
        positions = json.load(f)
    start_pos = positions[0]  # Low-Z central: x=143944, y=61076, z=192

    env = Environment(
        headless=False,
        config_path="config.json",
        verbose=False,
        reward_function=lambda s, a, ps: (0, False),
    )
    env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)

    # Use env's built-in state management to set starting position
    print(f"  Setting starting position: {start_pos['name']} ({start_pos['x']}, {start_pos['y']}, {start_pos['z']})")
    state_json = env.get_JSON_state()
    state = json.loads(state_json) if isinstance(state_json, str) else state_json
    state["position"] = [float(start_pos["x"]), float(start_pos["y"]), float(start_pos["z"])]
    env.change_JSON_state_url(state)

    print("  Waiting 8s for render...")
    time.sleep(8)

    # Re-read state after navigation
    state_json = env.get_JSON_state()
    state = json.loads(state_json) if isinstance(state_json, str) else state_json
    env.prev_json = state  # sync env's internal state
    env.prev_state = env.prepare_state()[0]
    init_pos = state["position"]
    print(f"  Initial position: {init_pos}")

    # Test 1: Move z by +50
    print("\n  Test 1: delta_z=50")
    vlm_text = '{"delta_x": 0, "delta_y": 0, "delta_z": 50}'
    parsed = parse_vlm_response(vlm_text)
    vec = vlm_json_to_action_vector(parsed)
    env.step(vec)
    time.sleep(1)
    state = json.loads(env.get_JSON_state()) if isinstance(env.get_JSON_state(), str) else env.get_JSON_state()
    new_pos = state["position"]
    dz = new_pos[2] - init_pos[2]
    print(f"    Position after: {new_pos}")
    print(f"    Z delta: {dz:.1f} (expected ~50)")
    assert abs(dz - 50) < 1, f"Z delta wrong: {dz}"
    print("    OK")

    # Test 2: Move x by +100, y by -50
    print("\n  Test 2: delta_x=100, delta_y=-50")
    prev_pos = new_pos[:]
    vec = vlm_json_to_action_vector({"delta_x": 100, "delta_y": -50, "delta_z": 0})
    env.step(vec)
    time.sleep(1)
    state = json.loads(env.get_JSON_state()) if isinstance(env.get_JSON_state(), str) else env.get_JSON_state()
    new_pos = state["position"]
    dx = new_pos[0] - prev_pos[0]
    dy = new_pos[1] - prev_pos[1]
    print(f"    Position after: {new_pos}")
    print(f"    X delta: {dx:.1f} (expected ~100), Y delta: {dy:.1f} (expected ~-50)")
    assert abs(dx - 100) < 1 and abs(dy - (-50)) < 1, f"Deltas wrong: dx={dx}, dy={dy}"
    print("    OK")

    # Test 3: Parse from markdown, move z by +25
    print("\n  Test 3: parse markdown -> delta_z=25")
    prev_z = new_pos[2]
    vlm_text = 'I will move up.\n```json\n{"delta_x": 0, "delta_y": 0, "delta_z": 25}\n```'
    parsed = parse_vlm_response(vlm_text)
    vec = vlm_json_to_action_vector(parsed)
    env.step(vec)
    time.sleep(1)
    state = json.loads(env.get_JSON_state()) if isinstance(env.get_JSON_state(), str) else env.get_JSON_state()
    new_z = state["position"][2]
    dz = new_z - prev_z
    print(f"    Z after: {new_z:.1f}, delta: {dz:.1f} (expected ~25)")
    assert abs(dz - 25) < 1, f"Z delta wrong: {dz}"
    print("    OK")

    print("\n  ALL LIVE TESTS PASSED")
    env.end_session()


if __name__ == "__main__":
    # Run unit tests first (no browser needed)
    test_parse_vlm_response()
    test_vlm_json_to_action_vector()
    test_starting_positions()

    # Run live test if --live flag passed
    if "--live" in sys.argv:
        test_env_roundtrip()
    else:
        print("Skipping live env test. Run with --live to include it.")
