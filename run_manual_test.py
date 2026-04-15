"""
Manual test entry point for the VLM agent.

Runs a single episode: one starting position, N steps, one VLM model.
Prints each step's state, VLM response, and action taken.
Saves screenshots to results/manual_test/.

Usage:
    # Default: Claude Sonnet, position 1, 14 steps
    python run_manual_test.py

    # GPT-4o, positions 3 and 4, 20 steps each
    python run_manual_test.py --model gpt-4o --position 3 4 --steps 20

    # Slower EM tile load wait
    python run_manual_test.py --model gpt-5 --position 1 --steps 30 --post-step-delay 4.0

    # Group outputs under results/manual_test/<run-id>/...
    python run_manual_test.py --model gpt-5 --run-id 2026-04-13_expA --steps 30

Requires API keys set as environment variables (or in .env file):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""

import argparse
import json
import os
import time
import urllib.parse

from dotenv import load_dotenv

load_dotenv()

from ngllib import Environment
from vlm_navigator.agents.vlm_agent import VLMAgent
from vlm_navigator.utils.nerve_visibility import (
    build_static_mask,
    classify_visibility,
    visibility_score,
)


# ---- Segment presets ----
# Segment ID -> human-readable label
SEGMENTS = {
    "720575940603464672": "original",
    "720575940615862389": "segB",
    "720575940630914858": "segC",
    "720575940623617973": "segD",
}

_NGL_BASE_STATE = {
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "crossSectionScale": 2.0339912586467497,
    "projectionOrientation": [-0.4705163836479187, 0.8044001460075378, -0.30343097448349, 0.1987067461013794],
    "projectionScale": 13976.00585680798,
    "layers": [
        {
            "type": "image",
            "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
            "tab": "source",
            "name": "Maryland (USA)-image",
        },
        {
            "type": "segmentation",
            "source": "precomputed://gs://flywire_v141_m783",
            "tab": "source",
            "name": "flywire_v141_m783",
        },
    ],
    "showDefaultAnnotations": False,
    "selectedLayer": {"size": 350, "layer": "flywire_v141_m783"},
    "layout": "xy-3d",
}


def build_ngl_url(segment_id: str, position: list | None = None) -> str:
    """Build a Neuroglancer URL with only the given segment active."""
    import copy
    state = copy.deepcopy(_NGL_BASE_STATE)
    state["layers"][1]["segments"] = [segment_id]
    if position is not None:
        state["position"] = [float(position[0]), float(position[1]), float(position[2])]
    fragment = urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    return f"https://neuroglancer-demo.appspot.com/#!{fragment}"


# ---- Model presets ----
MODEL_PRESETS = {
    "claude-sonnet": {
        "model": "anthropic/claude-sonnet-4-20250514",
        "max_tokens": 300,
        "temperature": 0.0,
    },
    "gpt-4o": {
        "model": "gpt-4o",
        "max_tokens": 300,
        "temperature": 0.0,
    },
    "gpt-5": {
        "model": "gpt-5",
        "max_tokens": 2048,
        "temperature": 1.0,
    },
    "gemini-flash": {
        "model": "gemini/gemini-2.0-flash",
        "max_tokens": 300,
        "temperature": 0.0,
    },
    "gemini-pro": {
        "model": "gemini/gemini-2.5-pro",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}

'''
SYSTEM_PROMPT = """You  are navigating a 3D brain electron microscopy (EM) dataset in Neuroglancer to find the highest-Z tip of a specific neuron segment.

  PANELS:
  - LEFT: 2D cross-section of current Z slice. Grey = raw EM tissue. Colored overlay = your target neuron segment. Crosshair = current position.
  - RIGHT: 3D mesh projection of the neuron so far.

  YOUR TASK: Reach the axon terminus — the highest Z position where the colored segment still exists. You are done when the segment disappears and cannot be recovered by x/y search.

  STRATEGY:
  1. Consult the 3D mesh (right panel) to understand where the nerve continues in 3D space and guide your x/y positioning.
  2. As a general guide: off-center neuron → bold x/y correction (250-1000 units); centered neuron → advance z (50-150 units); no neuron → search x/y (250-2000 units) before retreating z.
  4. If NO visible neuron (black/empty): you overshot or drifted. DECREASE delta_z (go backward) or adjust x/y to find the neuron again
  
  SCALE REFERENCE:
  - The 750nm scale bar in the bottom-left of the 2D panel is approximately 1/8th of the panel width.
  - x/y steps of 10-20 are almost invisible. Use 50-200 for meaningful repositioning.
  - z steps of 5-10 are wastefully small when the neuron is clearly visible. Default to 50+.

  COORDINATE SYSTEM:
  - Y-axis points DOWN — increasing delta_y moves the view downward on screen.
  - To move toward something above centre, use a negative delta_y.
  - To move toward something below centre, use a positive delta_y.

  IMPORTANT:
  - The neuron curves through 3D space — it won't stay at the same x,y as you change z
  - Small z steps (10-30) are safer than large ones (50-100) when the neuron is near an edge
  - Re-centering the neuron (x/y adjustment) is MORE important than z progress
  - You may end the run early if you are confident further z-progress is impossible
    (e.g. the nerve has been lost for multiple steps and x/y corrections have failed).
    If ending, include "done": true in the same JSON object alongside the deltas.

  Respond with a brief observation (1 sentence) then ONLY a JSON object:
  {"delta_x": 0, "delta_y": 0, "delta_z": 50}
"""

'''

SYSTEM_PROMPT = """You  are navigating a 3D brain electron microscopy (EM) dataset in Neuroglancer to find the highest-Z tip of a specific neuron segment.

  PANELS:
  - LEFT: 2D cross-section of current Z slice. Grey = raw EM tissue. Colored overlay = your target neuron segment. Crosshair = current position.
  - RIGHT: 3D mesh projection of the neuron so far.

  YOUR TASK: Reach the axon terminus — the highest Z position where the colored segment still exists. You are done when the segment disappears and cannot be recovered by x/y search.

  STRATEGY:
  1. LOOK at the screenshot. Is colored neuron tissue visible?
  2. If the neuron is clearly visible and roughly centered: advance with delta_z = 50-150. Only use smaller z steps (10-30) if the neuron is near the edge of the frame or looks like it's about to exit.
  3. If the neuron is visible but off-center: make a BOLD x/y correction (250-1000 units)
  4. If NO visible neuron (black/empty): you overshot or drifted. DECREASE delta_z (go backward) or adjust x/y to find the neuron again
  5. If the neuron shifts left/right/up/down between steps, compensate with x/y deltas
  6. Always consult the 3D mesh (right panel) to determine where the nerve continues in 3D space, and use this to guide your x/y positioning.
  
  SCALE REFERENCE:
  - The 750nm scale bar in the bottom-left of the 2D panel is approximately 1/8th of the panel width.
  - x/y steps of 10-20 are almost invisible. Use 50-200 for meaningful repositioning.
  - z steps of 5-10 are wastefully small when the neuron is clearly visible. Default to 50+.

  COORDINATE SYSTEM:
  - Y-axis points DOWN — increasing delta_y moves the view downward on screen.
  - To move toward something above centre, use a negative delta_y.
  - To move toward something below centre, use a positive delta_y.

  IMPORTANT:
  - The neuron curves through 3D space — it won't stay at the same x,y as you change z
  - Small z steps (10-30) are safer than large ones (50-100) when the neuron is near an edge
  - Re-centering the neuron (x/y adjustment) is MORE important than z progress
  - You may end the run early if you are confident further z-progress is impossible
    (e.g. the nerve has been lost for multiple steps and x/y corrections have failed).
    If ending, include "done": true in the same JSON object alongside the deltas.

  Respond with a brief observation (1 sentence) then ONLY a JSON object:
  {"delta_x": 0, "delta_y": 0, "delta_z": 50}
"""


STEP_PROMPT_TEMPLATE = (
    "Step {step}/{max_steps}. "
    "Position: [{pos_x:.0f}, {pos_y:.0f}, {pos_z:.0f}]. "
    "Z delta from last step: {prev_z_delta:+.0f}. "
    "Current Z: {current_z:.0f}. "
    "Nerve visible this step: {nerve_visible}. "
    "Recent history (oldest→newest): {action_history}. "
    "Screenshot has 2D and 3D views. "
    "Return the next action as JSON."
)


def _safe_run_id(run_id: str) -> str:
    rid = run_id.strip()
    if not rid:
        raise ValueError("--run-id must be non-empty after stripping")
    if ".." in rid:
        raise ValueError("--run-id must not contain '..'")
    return rid.replace(os.sep, "_").replace("/", "_").replace("\\", "_")


def _output_dir(run_id: str | None, model_name: str, position_id: int, trial: int) -> str:
    base = os.path.join("results", "manual_test")
    if run_id is not None:
        base = os.path.join(base, _safe_run_id(run_id))
    return os.path.join(base, f"{model_name}_pos{position_id}_trial{trial:02d}")


def make_reward_fn(nerve_visible_ref: list):
    """Reward function using nerve visibility from the current step.

    nerve_visible_ref is a 1-element list updated in-place before env.step().

    Reward logic:
      - visible     : +z_delta
      - uncertain   : +0.5 * z_delta
      - not_visible : -1 flat penalty
    """
    def reward_fn(state, action, prev_state):
        label = nerve_visible_ref[0]
        z_delta = state[0][0][2] - prev_state[0][0][2]
        if label == "visible":
            return float(z_delta), False
        elif label == "uncertain":
            return 0.5 * float(z_delta), False
        else:
            return -1.0, False
    return reward_fn


def run_manual_test(
    model_name: str,
    position_id: int,
    max_steps: int,
    save_debug: bool = False,
    stop_mode: str = "agent",
    trial: int = 1,
    min_steps_before_stop: int = 8,
    post_step_delay: float = 3.0,
    run_id: str | None = None,
    segment_id: str | None = None,
):
    with open("vlm_navigator/config/starting_positions.json") as f:
        positions = json.load(f)

    start_pos = next((p for p in positions if p["id"] == position_id), None)
    if not start_pos:
        print(f"Position ID {position_id} not found. Available: {[p['id'] for p in positions]}")
        return

    # Determine segment: explicit arg > position's segment_id > fallback to original
    effective_segment = (
        segment_id
        or start_pos.get("segment_id")
        or "720575940603464672"
    )
    segment_label = SEGMENTS.get(effective_segment, effective_segment[-6:])

    model_config = MODEL_PRESETS.get(model_name)
    if not model_config:
        print(f"Unknown model '{model_name}'. Available: {list(MODEL_PRESETS.keys())}")
        return

    output_dir = _output_dir(run_id, f"{model_name}_{segment_label}", position_id, trial)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Model:    {model_name} ({model_config['model']})")
    print(f"Segment:  {effective_segment} ({segment_label})")
    print(f"Position: #{start_pos['id']} {start_pos['name']} "
          f"({start_pos['x']}, {start_pos['y']}, {start_pos['z']})")
    print(f"StopMode: {stop_mode}")
    print(f"Trial:    {trial}")
    print(f"Steps:    {max_steps}")
    print(f"PostStep: {post_step_delay}s")
    if run_id:
        print(f"Run ID:   {run_id}")
    print(f"Output:   {output_dir}/")
    print("=" * 60)

    nerve_visible_ref = ["uncertain"]
    reward_fn = make_reward_fn(nerve_visible_ref)

    # Init environment (retry up to 3x — Neuroglancer can occasionally return empty state)
    env = None
    last_start_error = None
    for attempt in range(1, 4):
        try:
            ngl_url = build_ngl_url(
                effective_segment,
                [start_pos["x"], start_pos["y"], start_pos["z"]],
            )
            env = Environment(
                headless=False,
                config_path="config.json",
                verbose=False,
                reward_function=reward_fn,
                start_url=ngl_url,
            )
            env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)
            if not env.get_JSON_state():
                raise RuntimeError("Environment returned empty JSON state after start_session")
            break
        except Exception as e:
            last_start_error = e
            print(f"  Startup attempt {attempt}/3 failed: {e}")
            if env is not None:
                try:
                    env.end_session()
                except Exception:
                    pass
            env = None
            time.sleep(3)
    if env is None:
        raise RuntimeError(f"Failed to initialize environment after retries: {last_start_error}")

    # Navigate to starting position
    print("Setting starting position...")
    time.sleep(5)
    state_str = env.get_JSON_state()
    if not state_str:
        raise RuntimeError("get_JSON_state() returned empty/None while setting starting position")
    json_state = json.loads(state_str) if isinstance(state_str, str) else state_str
    json_state["position"] = [float(start_pos["x"]), float(start_pos["y"]), float(start_pos["z"])]
    # Re-assert the correct segment — get_JSON_state() may return a stale browser state
    # (e.g. from a previous run) that has the wrong segment active.
    for layer in json_state.get("layers", []):
        if layer.get("type") == "segmentation":
            layer["segments"] = [effective_segment]
            break
    env.change_JSON_state_url(json_state)
    time.sleep(5)
    env.prev_state, env.prev_json = env.prepare_state()

    # Init agent
    agent = VLMAgent(model_config, action_mode="position_only")
    agent.reset(SYSTEM_PROMPT)

    # Episode loop
    prev_z_delta = 0.0
    start_z = start_pos["z"]
    log = []
    consecutive_not_visible = 0
    action_history: list[dict] = []
    saved_screenshot_paths: list[str] = []  # for building static mask
    static_mask = None  # updated each step once we have enough frames

    print(f"\nStarting episode from z={start_z}...\n")

    for step in range(1, max_steps + 1):
        (pos_state, screenshot), cur_json = env.prev_state, env.prev_json
        position = pos_state[0]
        cross_section_scale = pos_state[1]
        orientation = pos_state[2]
        projection_scale = pos_state[3]

        # Save screenshot and check nerve visibility
        screenshot_path = os.path.join(output_dir, f"step_{step:03d}.jpg")
        screenshot.save(screenshot_path, format="JPEG")
        saved_screenshot_paths.append(screenshot_path)

        # Rebuild static mask from all frames so far (filters static UI green artifacts).
        # With only 1 frame the mask would be the frame itself (masking all green), so
        # we wait until step 2 before applying it.
        if len(saved_screenshot_paths) >= 2:
            static_mask = build_static_mask(saved_screenshot_paths)

        vis_score = visibility_score(screenshot_path, static_mask=static_mask)
        nerve_label = classify_visibility(vis_score["dynamic_colored_fraction"])

        if nerve_label == "not_visible":
            consecutive_not_visible += 1
        else:
            consecutive_not_visible = 0

        if consecutive_not_visible >= 3:
            print(f"\n  3 consecutive steps with no nerve visible (step {step}). Ending episode early.")
            log.append({
                "step": step,
                "position_before": list(position),
                "action": None,
                "raw_response": None,
                "position_after": list(position),
                "z_delta": 0.0,
                "nerve_visible": nerve_label,
                "early_stop": True,
            })
            break

        # Build step prompt
        history_str = "; ".join(
            f"step {h['step']}: dx={h['delta_x']}, dy={h['delta_y']}, dz={h['delta_z']}, nerve={h['nerve']}"
            for h in action_history
        ) if action_history else "none yet"

        step_text = STEP_PROMPT_TEMPLATE.format(
            step=step,
            max_steps=max_steps,
            pos_x=position[0],
            pos_y=position[1],
            pos_z=position[2],
            prev_z_delta=prev_z_delta,
            current_z=position[2],
            nerve_visible=nerve_label,
            action_history=history_str,
        )

        if save_debug:
            screenshot.resize((960, 540)).save(
                os.path.join(debug_dir, f"step_{step:03d}_vlm_input.jpg"), format="JPEG"
            )
            with open(os.path.join(debug_dir, f"step_{step:03d}_prompt.txt"), "w") as df:
                df.write(f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== STEP PROMPT ===\n{step_text}\n")

        # Get VLM action
        try:
            action_vector, parsed, raw_text = agent.get_action(
                screenshot=screenshot,
                position=position,
                orientation=orientation,
                cross_section_scale=cross_section_scale,
                projection_scale=projection_scale,
                prev_z_delta=prev_z_delta,
                step_prompt_template=step_text,
            )
        except Exception as e:
            print(f"  Step {step}: VLM call failed: {e}")
            raw_text = "ERROR"
            parsed = {"delta_x": 0, "delta_y": 0, "delta_z": 5}
            action_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 0, 0, 0, 0]

        if save_debug:
            with open(os.path.join(debug_dir, f"step_{step:03d}_response.txt"), "w") as df:
                df.write(f"=== RAW RESPONSE ===\n{raw_text}\n\n=== PARSED ACTION ===\n{json.dumps(parsed, indent=2)}\n")

        # Step environment
        nerve_visible_ref[0] = nerve_label
        prev_z = position[2]
        state, reward, done, new_json = env.step(action_vector)
        time.sleep(post_step_delay)
        env.prev_state, env.prev_json = env.prepare_state()

        new_position = state[0][0]
        prev_z_delta = new_position[2] - prev_z

        log.append({
            "step": step,
            "position_before": list(position),
            "action": parsed,
            "raw_response": raw_text,
            "position_after": list(new_position),
            "z_delta": prev_z_delta,
            "nerve_visible": nerve_label,
        })

        action_history.append({
            "step": step,
            "delta_x": parsed.get("delta_x", 0),
            "delta_y": parsed.get("delta_y", 0),
            "delta_z": parsed.get("delta_z", 0),
            "nerve": nerve_label,
        })
        if len(action_history) > 3:
            action_history.pop(0)

        print(f"  Step {step:3d} | z: {prev_z:.0f} -> {new_position[2]:.0f} "
              f"(dz: {prev_z_delta:+.0f}) | nerve: {nerve_label} | action: {parsed}")

        if step % 10 == 0:
            agent.trim_history(keep_last_n=5)

        if new_position[2] >= 6500:
            print(f"\n  Reached z={new_position[2]:.0f} >= 6500. Done!")
            break
        if stop_mode == "agent" and step >= min_steps_before_stop and bool(parsed.get("done", False)):
            print(f"\n  Agent requested stop at step {step}. Done!")
            break

    # Save final screenshot and log
    env.get_screenshot().save(os.path.join(output_dir, "final.jpg"), format="JPEG")
    with open(os.path.join(output_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)

    final_z = log[-1]["position_after"][2] if log else start_z
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  Start Z:        {start_z}")
    print(f"  Final Z:        {final_z:.0f}")
    print(f"  Z gained:       {final_z - start_z:.0f}")
    print(f"  Steps taken:    {len(log)}")
    print(f"  Parse failures: {agent.parse_failures}")
    print(f"  Log saved to:   {output_dir}/log.json")
    print(f"{'=' * 60}")

    env.end_session()
    return {
        "model": model_name,
        "position_id": position_id,
        "trial": trial,
        "steps_taken": len(log),
        "start_z": start_z,
        "final_z": final_z,
        "z_gained": final_z - start_z,
        "parse_failures": agent.parse_failures,
        "stop_mode": stop_mode,
        "run_id": run_id,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual VLM agent test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python run_manual_test.py --model gpt-5 --position 1

  # Batch: positions 3 and 4, 2 trials each
  python run_manual_test.py --model gpt-5 --position 3 4 --trials 2

  # Batch: all positions for a model
  python run_manual_test.py --model claude-sonnet --position all --trials 2
""",
    )
    parser.add_argument("--model", default="claude-sonnet", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument(
        "--position",
        nargs="+",
        metavar="POS",
        default=None,
        help="One or more starting position IDs (e.g. --position 3 4), or 'all'. Default: 1.",
    )
    parser.add_argument("--steps", type=int, default=14, help="Max steps per episode")
    parser.add_argument(
        "--stop-mode",
        default="agent",
        choices=["fixed", "agent"],
        help="fixed=run full step budget, agent=allow model to stop early with done=true (default)",
    )
    parser.add_argument(
        "--min-steps-before-stop",
        type=int,
        default=8,
        help="Minimum steps before accepting agent stop request",
    )
    parser.add_argument(
        "--post-step-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after each env.step() for EM tiles to load (default 3.0)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help="Optional folder name under results/manual_test/ to group outputs",
    )
    parser.add_argument("--debug", action="store_true", help="Save VLM input images, prompts, and responses to debug/")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run sequentially")
    parser.add_argument(
        "--segment",
        default=None,
        choices=list(SEGMENTS.keys()) + list(SEGMENTS.values()),
        metavar="SEG",
        help=(
            "Segment ID or label to display. "
            f"Labels: {', '.join(SEGMENTS.values())}. "
            "Overrides the segment_id in starting_positions.json."
        ),
    )
    args = parser.parse_args()

    # Resolve label -> segment ID if the user passed a label like "segB"
    label_to_id = {v: k for k, v in SEGMENTS.items()}
    raw_seg = args.segment
    resolved_segment = label_to_id.get(raw_seg, raw_seg)  # None if not passed

    # Resolve position list
    with open("vlm_navigator/config/starting_positions.json") as _f:
        _all_positions = [p["id"] for p in json.load(_f)]

    if args.position is not None:
        if args.position == ["all"]:
            position_ids = _all_positions
        else:
            position_ids = [int(p) for p in args.position]
    else:
        position_ids = [1]  # default

    total = len(position_ids) * args.trials
    done = 0
    failed = 0
    for position_id in position_ids:
        for trial in range(1, args.trials + 1):
            done += 1
            print(f"\n[Batch {done}/{total}] position={position_id} trial={trial}")
            try:
                run_manual_test(
                    model_name=args.model,
                    position_id=position_id,
                    max_steps=args.steps,
                    save_debug=args.debug,
                    stop_mode=args.stop_mode,
                    trial=trial,
                    min_steps_before_stop=args.min_steps_before_stop,
                    post_step_delay=args.post_step_delay,
                    run_id=args.run_id,
                    segment_id=resolved_segment,
                )
            except Exception as e:
                failed += 1
                print(f"\n  [ERROR] position={position_id} trial={trial} failed: {e}")
                print(f"  Continuing with remaining runs...\n")

    if failed:
        print(f"\n{failed}/{total} run(s) failed.")
