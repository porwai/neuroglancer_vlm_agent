"""
Manual test entry point for the VLM agent.

Runs a single episode: one starting position, N steps, one VLM model.
Prints each step's state, VLM response, and action taken.
Saves screenshots to results/manual_test/.

Usage:
    # Default: Claude Sonnet, position 1, 14 steps
    python run_manual_test.py

    # GPT-4o, position 8 (Mid-Z central), minimal prompt, 20 steps
    python run_manual_test.py --model gpt-4o --position 8 --steps 20 --prompt-variant minimal

    # Prompt ablation across all variants with 3 trials each
    python run_manual_test.py --ablate --trials 3 --steps 30

Requires API keys set as environment variables (or in .env file):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

from ngllib import Environment
from vlm_navigator.agents.vlm_agent import VLMAgent

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
}

# ---- Prompt ablation variants (kept in this file on purpose) ----
PROMPT_VARIANTS = {
    "present": """You are navigating a 3D brain electron microscopy (EM) dataset in Neuroglancer.
  The view shows a 2D cross-section (left panel) and a 3D projection (right panel).

  WHAT YOU SEE:
  - Grey texture = raw EM brain tissue
  - Colored overlay = segmented neuron (your target to follow)
  - Black/empty = outside the dataset or no tissue at this position

  YOUR TASK: Follow the colored neuron segment through 3D space toward higher Z positions.

  STRATEGY:
  1. LOOK at the screenshot. Is colored neuron tissue visible?
  2. If the neuron is clearly visible and roughly centered: advance with delta_z = 50-150. Only use smaller z steps (10-30) if the neuron is near the edge of the frame or looks like it's about to exit.
  3. If the neuron is visible but off-center: make a BOLD x/y correction (250-1000 units)
  4. If NO visible neuron (black/empty): you overshot or drifted. DECREASE delta_z (go backward) or adjust x/y to find the neuron again
  5. If the neuron shifts left/right/up/down between steps, compensate with x/y deltas

  SCALE REFERENCE:
  - x/y steps of 10-20 are almost invisible. Use 50-200 for meaningful repositioning.
  - z steps of 5-10 are wastefully small when the neuron is clearly visible. Default to 50+.

  IMPORTANT:
  - The neuron curves through 3D space — it won't stay at the same x,y as you change z
  - Small z steps (10-30) are safer than large ones (50-100) when the neuron is near an edge, if according to the right view, there is alot of room to inc
  - Re-centering the neuron (x/y adjustment) is MORE important than z progress
  - Note that there is a size guide of 750nm in the bottom left, this means that step size for x and y can be relatively large

  Respond with a brief observation (1 sentence) then ONLY a JSON object:
  {"delta_x": 0, "delta_y": 0, "delta_z": 50}
""",
    "minimal": """You are navigating a 3D brain EM dataset in Neuroglancer.
The screenshot includes a 2D cross-section and a 3D view.
Goal: follow the same colored neuron and maximize z while staying on that neuron.
Use the screenshot and current state to choose the next move.
Return ONLY JSON:
{"delta_x": number, "delta_y": number, "delta_z": number}
""",
    "balanced": """You are navigating a 3D brain EM dataset in Neuroglancer.
The screenshot includes a 2D cross-section (left) and a 3D view (right).
Goal: follow the same colored neuron and reach as high a z as possible while staying on that neuron.
If the neuron appears off-center, adjust x/y to keep tracking it. If it is centered and visible, continue progressing in z.
Return ONLY JSON:
{"delta_x": number, "delta_y": number, "delta_z": number}
""",
    "guided": """You are navigating a 3D brain electron microscopy (EM) dataset in Neuroglancer.
The view shows a 2D cross-section (left panel) and a 3D projection (right panel).

WHAT YOU SEE:
- Grey texture = raw EM brain tissue
- Colored overlay = segmented neuron (your target to follow)
- Black/empty = outside the dataset or no tissue at this position

YOUR TASK: Follow the colored neuron segment through 3D space toward higher Z positions.

STRATEGY:
1. LOOK at the screenshot. Is colored neuron tissue visible?
2. If the neuron is clearly visible and roughly centered: advance with delta_z = 50-150.
3. If the neuron is visible but off-center: make a BOLD x/y correction (250-1000 units)
4. If NO visible neuron (black/empty): you overshot or drifted. DECREASE delta_z (go backward) or adjust x/y to find the neuron again
5. If the neuron shifts left/right/up/down between steps, compensate with x/y deltas

IMPORTANT:
- The neuron curves through 3D space — it won't stay at the same x,y as you change z
- Re-centering the neuron (x/y adjustment) can be more important than z progress

Return ONLY JSON:
{"delta_x": number, "delta_y": number, "delta_z": number}
""",
}

STEP_PROMPT_TEMPLATE = (
    "Step {step}/{max_steps}. "
    "Position: [{pos_x:.0f}, {pos_y:.0f}, {pos_z:.0f}]. "
    "Z delta from last step: {prev_z_delta:+.0f}. "
    "Current Z: {current_z:.0f}. "
    "Screenshot has 2D and 3D views. "
    "Return the next action as JSON."
)


def build_system_prompt(prompt_variant: str, allow_agent_stop: bool) -> str:
    base_prompt = PROMPT_VARIANTS[prompt_variant]
    if not allow_agent_stop:
        return base_prompt
    return (
        base_prompt
        + '\nYou may end the run if you are confident further progress is unlikely.'
        + '\nIf ending, include "done": true in the same JSON object.'
    )

def run_manual_test(
    model_name: str,
    position_id: int,
    max_steps: int,
    save_debug: bool = False,
    prompt_variant: str = "minimal",
    stop_mode: str = "fixed",
    trial: int = 1,
    min_steps_before_stop: int = 8,
):
    # Load starting positions
    with open("vlm_navigator/config/starting_positions.json") as f:
        positions = json.load(f)

    start_pos = next((p for p in positions if p["id"] == position_id), None)
    if not start_pos:
        print(f"Position ID {position_id} not found. Available: {[p['id'] for p in positions]}")
        return

    model_config = MODEL_PRESETS.get(model_name)
    if not model_config:
        print(f"Unknown model '{model_name}'. Available: {list(MODEL_PRESETS.keys())}")
        return

    # Setup output dir
    output_dir = f"results/manual_test/{model_name}_pos{position_id}_{prompt_variant}_trial{trial:02d}"
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Model:    {model_name} ({model_config['model']})")
    print(f"Position: #{start_pos['id']} {start_pos['name']} "
          f"({start_pos['x']}, {start_pos['y']}, {start_pos['z']})")
    print(f"Prompt:   {prompt_variant}")
    print(f"StopMode: {stop_mode}")
    print(f"Trial:    {trial}")
    print(f"Steps:    {max_steps}")
    print(f"Output:   {output_dir}/")
    print("=" * 60)

    # Init environment (Neuroglancer startup can occasionally return empty state; retry)
    env = None
    last_start_error = None
    for attempt in range(1, 4):
        try:
            env = Environment(
                headless=False,
                config_path="config.json",
                verbose=False,
                reward_function=lambda s, a, ps: (0, False),
            )
            env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)
            state_probe = env.get_JSON_state()
            if not state_probe:
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

    # Navigate to starting position using env's built-in state management
    print("Setting starting position...")
    time.sleep(5)
    state_str = env.get_JSON_state()
    if not state_str:
        raise RuntimeError("get_JSON_state() returned empty/None while setting starting position")
    json_state = json.loads(state_str) if isinstance(state_str, str) else state_str
    json_state["position"] = [float(start_pos["x"]), float(start_pos["y"]), float(start_pos["z"])]
    env.change_JSON_state_url(json_state)
    time.sleep(5)

    # Sync env internal state after URL change
    env.prev_state, env.prev_json = env.prepare_state()

    # Init agent
    agent = VLMAgent(model_config, action_mode="position_only")
    system_prompt = build_system_prompt(
        prompt_variant=prompt_variant,
        allow_agent_stop=(stop_mode == "agent"),
    )
    agent.reset(system_prompt)

    # Episode loop
    prev_z_delta = 0.0
    start_z = start_pos["z"]
    log = []

    print(f"\nStarting episode from z={start_z}...\n")

    for step in range(1, max_steps + 1):
        # Get current state
        (pos_state, screenshot), cur_json = env.prev_state, env.prev_json
        position = pos_state[0]
        cross_section_scale = pos_state[1]
        orientation = pos_state[2]
        projection_scale = pos_state[3]

        # Save screenshot
        screenshot.save(os.path.join(output_dir, f"step_{step:03d}.jpg"), format="JPEG")

        # Build step prompt
        step_text = STEP_PROMPT_TEMPLATE.format(
            step=step,
            max_steps=max_steps,
            pos_x=position[0],
            pos_y=position[1],
            pos_z=position[2],
            prev_z_delta=prev_z_delta,
            current_z=position[2],
        )

        # Save debug info: the exact image + prompt the agent sees
        if save_debug:
            resized = screenshot.resize((960, 540))
            resized.save(os.path.join(debug_dir, f"step_{step:03d}_vlm_input.jpg"), format="JPEG")
            with open(os.path.join(debug_dir, f"step_{step:03d}_prompt.txt"), "w") as df:
                df.write(f"=== SYSTEM PROMPT ({prompt_variant}) ===\n{system_prompt}\n\n")
                df.write(f"=== STEP PROMPT ===\n{step_text}\n")

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

        # Save VLM response to debug
        if save_debug:
            with open(os.path.join(debug_dir, f"step_{step:03d}_response.txt"), "w") as df:
                df.write(f"=== RAW RESPONSE ===\n{raw_text}\n\n")
                df.write(f"=== PARSED ACTION ===\n{json.dumps(parsed, indent=2)}\n")

        # Step the environment
        prev_z = position[2]
        state, reward, done, new_json = env.step(action_vector)
        time.sleep(0.5)

        new_position = state[0][0]
        prev_z_delta = new_position[2] - prev_z

        # Log
        entry = {
            "step": step,
            "position_before": list(position),
            "action": parsed,
            "raw_response": raw_text,
            "position_after": list(new_position),
            "z_delta": prev_z_delta,
        }
        log.append(entry)

        print(f"  Step {step:3d} | z: {prev_z:.0f} -> {new_position[2]:.0f} "
              f"(dz: {prev_z_delta:+.0f}) | action: {parsed}")

        # Trim history every 10 steps to avoid context overflow
        if step % 10 == 0:
            agent.trim_history(keep_last_n=5)

        # Check done condition
        if new_position[2] >= 6500:
            print(f"\n  Reached z={new_position[2]:.0f} >= 6500. Done!")
            break
        if (
            stop_mode == "agent"
            and step >= min_steps_before_stop
            and bool(parsed.get("done", False))
        ):
            print(f"\n  Agent requested stop at step {step}. Done!")
            break

    # Save final screenshot
    final_screenshot = env.get_screenshot()
    final_screenshot.save(os.path.join(output_dir, "final.jpg"), format="JPEG")

    # Save log
    with open(os.path.join(output_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)

    # Summary
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
        "prompt_variant": prompt_variant,
        "trial": trial,
        "steps_taken": len(log),
        "start_z": start_z,
        "final_z": final_z,
        "z_gained": final_z - start_z,
        "parse_failures": agent.parse_failures,
        "stop_mode": stop_mode,
    }


def run_prompt_ablation(
    model_name: str,
    position_id: int,
    max_steps: int,
    trials: int,
    save_debug: bool,
    stop_mode: str,
    min_steps_before_stop: int,
):
    print("\nRunning prompt ablation...")
    print(f"Variants: {list(PROMPT_VARIANTS.keys())}")
    print(f"Trials per variant: {trials}")
    print("=" * 60)

    all_results = []
    for variant in PROMPT_VARIANTS:
        for trial in range(1, trials + 1):
            result = run_manual_test(
                model_name=model_name,
                position_id=position_id,
                max_steps=max_steps,
                save_debug=save_debug,
                prompt_variant=variant,
                stop_mode=stop_mode,
                trial=trial,
                min_steps_before_stop=min_steps_before_stop,
            )
            all_results.append(result)

    print("\nABLATION SUMMARY")
    print("=" * 60)
    for variant in PROMPT_VARIANTS:
        rows = [r for r in all_results if r["prompt_variant"] == variant]
        avg_final_z = sum(r["final_z"] for r in rows) / len(rows)
        avg_gain = sum(r["z_gained"] for r in rows) / len(rows)
        avg_steps = sum(r["steps_taken"] for r in rows) / len(rows)
        print(
            f"{variant:>8} | trials={len(rows):2d} | "
            f"avg_final_z={avg_final_z:7.1f} | avg_gain={avg_gain:7.1f} | avg_steps={avg_steps:5.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual VLM agent test")
    parser.add_argument("--model", default="claude-sonnet", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--position", type=int, default=1, help="Starting position ID (1-16)")
    parser.add_argument("--steps", type=int, default=14, help="Max steps per episode")
    parser.add_argument(
        "--prompt-variant",
        default="minimal",
        choices=list(PROMPT_VARIANTS.keys()),
        help="Prompt variant for single-run mode",
    )
    parser.add_argument(
        "--stop-mode",
        default="fixed",
        choices=["fixed", "agent"],
        help="fixed=run full step budget, agent=allow model to stop early with done=true",
    )
    parser.add_argument(
        "--min-steps-before-stop",
        type=int,
        default=8,
        help="Minimum steps before accepting agent stop request",
    )
    parser.add_argument("--ablate", action="store_true", help="Run all prompt variants")
    parser.add_argument("--trials", type=int, default=1, help="Trials per prompt variant in ablation mode")
    parser.add_argument("--debug", action="store_true", help="Save VLM input images, prompts, and responses to debug/ folder")
    args = parser.parse_args()

    if args.ablate:
        run_prompt_ablation(
            model_name=args.model,
            position_id=args.position,
            max_steps=args.steps,
            trials=args.trials,
            save_debug=args.debug,
            stop_mode=args.stop_mode,
            min_steps_before_stop=args.min_steps_before_stop,
        )
    else:
        run_manual_test(
            model_name=args.model,
            position_id=args.position,
            max_steps=args.steps,
            save_debug=args.debug,
            prompt_variant=args.prompt_variant,
            stop_mode=args.stop_mode,
            trial=1,
            min_steps_before_stop=args.min_steps_before_stop,
        )
