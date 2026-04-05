"""
Manual test entry point for the VLM agent.

Runs a single episode: one starting position, N steps, one VLM model.
Prints each step's state, VLM response, and action taken.
Saves screenshots to results/manual_test/.

Usage:
    # Default: Claude Sonnet, position 1, 10 steps
    python run_manual_test.py

    # GPT-4o, position 8 (Mid-Z central), 20 steps
    python run_manual_test.py --model gpt-4o --position 8 --steps 20

    # Gemini Flash
    python run_manual_test.py --model gemini-flash --steps 5

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

# ---- Placeholder prompts (replace with real ones in Phase 2.1) ----
SYSTEM_PROMPT = """You are navigating a 3D brain electron microscopy (EM) dataset in Neuroglancer.
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
  """

STEP_PROMPT_TEMPLATE = (                                                                                                                                  
    "Step {step}/{max_steps}. "
    "Position: [{pos_x:.0f}, {pos_y:.0f}, {pos_z:.0f}]. "                                                                                                 
    "Z delta from last step: {prev_z_delta:+.0f}. "                                                                                                       
    "Current Z: {current_z:.0f}. "                                                                                                      
    "Look at the screenshot: Is the colored neuron visible and centered? "                                                                                
    "What is your next action?"                                                                                                                           
)                                                                                                                                                         

def run_manual_test(model_name: str, position_id: int, max_steps: int, save_debug: bool = False):
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
    output_dir = f"results/manual_test/{model_name}_pos{position_id}"
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"Model:    {model_name} ({model_config['model']})")
    print(f"Position: #{start_pos['id']} {start_pos['name']} "
          f"({start_pos['x']}, {start_pos['y']}, {start_pos['z']})")
    print(f"Steps:    {max_steps}")
    print(f"Output:   {output_dir}/")
    print("=" * 60)

    # Init environment
    env = Environment(
        headless=False,
        config_path="config.json",
        verbose=False,
        reward_function=lambda s, a, ps: (0, False),
    )
    env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)

    # Navigate to starting position using env's built-in state management
    print("Setting starting position...")
    time.sleep(5)
    state_str = env.get_JSON_state()
    json_state = json.loads(state_str) if isinstance(state_str, str) else state_str
    json_state["position"] = [float(start_pos["x"]), float(start_pos["y"]), float(start_pos["z"])]
    env.change_JSON_state_url(json_state)
    time.sleep(5)

    # Sync env internal state after URL change
    env.prev_state, env.prev_json = env.prepare_state()

    # Init agent
    agent = VLMAgent(model_config, action_mode="position_only")
    agent.reset(SYSTEM_PROMPT)

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
                df.write(f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n")
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
            "position_before": position,
            "action": parsed,
            "raw_response": raw_text,
            "position_after": new_position,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual VLM agent test")
    parser.add_argument("--model", default="claude-sonnet", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--position", type=int, default=1, help="Starting position ID (1-16)")
    parser.add_argument("--steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--debug", action="store_true", help="Save VLM input images, prompts, and responses to debug/ folder")
    args = parser.parse_args()

    run_manual_test(args.model, args.position, args.steps, save_debug=args.debug)
