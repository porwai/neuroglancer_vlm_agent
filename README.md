# VLM Navigator

Benchmarks vision-language models (Claude, GPT-4o, Gemini) as zero-shot navigation agents in [Neuroglancer](https://github.com/google/neuroglancer) on the FlyWire Drosophila brain EM dataset.

**Task**: given a **low-Z** starting position on a segmented neuron, navigate through the 3D volume toward the highest possible Z coordinate while staying on the same neuron. Benchmarks only include Low-Z starts (one per segment); Mid/High-Z starts were dropped as they do not reflect the intended evaluation.

---

## How it works

Each step the agent receives a Neuroglancer screenshot (2D cross-section + 3D projection) and outputs `{"delta_x", "delta_y", "delta_z"}` to move the viewport. Nerve visibility is checked heuristically on each frame via green-pixel detection. The episode ends when:
- Max steps reached
- Z ≥ 6500 (dataset ceiling)
- 3 consecutive frames with no nerve visible (auto early-stop)
- The VLM emits `"done": true` (agent stop mode)

**Reward**: `+z_delta` if nerve visible, `+0.5 * z_delta` if uncertain, `-1` if not visible.

---

## Structure

```
vlm_navigator/
  agents/vlm_agent.py             # LiteLLM-backed agent (Claude, GPT-4o, Gemini)
  utils/action_utils.py           # VLM JSON → 17-element action vector
  utils/nerve_visibility.py       # Green-pixel heuristic for nerve detection
  config/starting_positions.json  # 4 Low-Z benchmark starts (one per segment)
run_manual_test.py                # Main entry point
summarize.py                      # Aggregate logs → CSV for plotting
```

---

## Running

```bash
python run_manual_test.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `claude-sonnet` | Model to use: `claude-sonnet`, `gpt-4o`, `gpt-5`, `gemini-flash` |
| `--position` | `1` | Starting position ID (1–4). See `config/starting_positions.json` for names/coords |
| `--steps` | `14` | Maximum steps per episode |
| `--trials` | `1` | Number of sequential trials to run with the same config |
| `--stop-mode` | `agent` | `agent`: VLM can emit `"done": true` to stop early. `fixed`: always run full step budget |
| `--min-steps-before-stop` | `8` | Earliest step at which an agent stop request is accepted |
| `--post-step-delay` | `3.0` | Seconds to wait after each move for EM tiles to fully render before capturing the next frame |
| `--run-id` | none | Any name (e.g. `baseline`, `april-exp`). Groups all outputs under `results/manual_test/<name>/` |
| `--debug` | off | Saves per-step VLM input images, prompts, and raw responses to a `debug/` subfolder |

### Examples

```bash
# Default: Claude Sonnet, position 1, 14 steps
python run_manual_test.py

# GPT-4o from SegD Low-Z, 20 steps
python run_manual_test.py --model gpt-4o --position 4 --steps 20

# 3 trials grouped under a named run
python run_manual_test.py --trials 3 --steps 30 --run-id baseline

# Disable agent early stop, give tiles more time to load
python run_manual_test.py --stop-mode fixed --post-step-delay 4.0 --debug

# All four segments, 3 trials each (12 runs total)
python run_manual_test.py --model gpt-5 --positions 1 2 3 4 --trials 3

# All positions, 2 trials
python run_manual_test.py --model claude-sonnet --positions all --trials 2

# Single run
python run_manual_test.py --model gpt-5 --position 1
```

Runs execute sequentially and print `[Batch N/total]` progress. Each run gets its own output dir (for example `results/manual_test/gpt-5_original_pos1_trial01/`).


---

## Output

Each run writes to `results/manual_test/[<run-id>/]<model>_pos<N>_trial<NN>/`:

| File | Description |
|------|-------------|
| `step_NNN.jpg` | Screenshot at each step |
| `final.jpg` | Final frame |
| `log.json` | Per-step record: position, action, z-delta, nerve visibility |
| `debug/` | *(if `--debug`)* Per-step VLM inputs, prompts, raw responses |

### log.json schema (per step)
```json
{
  "step": 1,
  "position_before": [x, y, z],
  "action": {"delta_x": 0, "delta_y": 0, "delta_z": 100},
  "raw_response": "...",
  "position_after": [x, y, z],
  "z_delta": 100,
  "nerve_visible": "visible | uncertain | not_visible"
}
```

---

## Graphing results

Run `summarize.py` to aggregate all logs into two CSVs:

```bash
python summarize.py
# → results/summary.csv   (one row per run)
# → results/steps.csv     (one row per step, for trajectory plots)

# Scope to a specific run group
python summarize.py --results-dir results/manual_test/baseline
```

**`summary.csv` columns**: `model`, `position_id`, `trial`, `steps_taken`, `start_z`, `final_z`, `z_gained`, `best_z_on_nerve`, `best_z_on_nerve_gained`, `early_stop_nerve`, `agent_stop`, `steps_visible`, `steps_uncertain`, `steps_not_visible`

**`steps.csv` columns**: `model`, `position_id`, `trial`, `step`, `z`, `z_delta`, `nerve_visible`, `early_stop`

Suggested plots:
- **Z trajectory**: `steps.csv` → line plot of `z` vs `step`, grouped by `model`
- **Z gained per model**: `summary.csv` → bar chart of `z_gained`, grouped by `model`
- **Nerve visibility**: `summary.csv` → stacked bar of `steps_visible / uncertain / not_visible`

---

## Neuroglancer links (Low-Z benchmark starts)

Same viewer state as `run_manual_test.build_ngl_url`: [neuroglancer-demo](https://neuroglancer-demo.appspot.com/), FlyWire `fafbv14` + `flywire_v141_m783`, `xy-3d`, one segment highlighted.

| ID | Segment | Neuroglancer |
|----|---------|--------------|
| 1 | Original (`720575940603464672`) | [open](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B4e-09%2C%22m%22%5D%2C%22y%22%3A%5B4e-09%2C%22m%22%5D%2C%22z%22%3A%5B4e-08%2C%22m%22%5D%7D%2C%22crossSectionScale%22%3A2.0339912586467497%2C%22projectionOrientation%22%3A%5B-0.4705163836479187%2C0.8044001460075378%2C-0.30343097448349%2C0.1987067461013794%5D%2C%22projectionScale%22%3A13976.00585680798%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A//https%3A//bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22precomputed%3A//gs%3A//flywire_v141_m783%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22flywire_v141_m783%22%2C%22segments%22%3A%5B%22720575940603464672%22%5D%7D%5D%2C%22showDefaultAnnotations%22%3Afalse%2C%22selectedLayer%22%3A%7B%22size%22%3A350%2C%22layer%22%3A%22flywire_v141_m783%22%7D%2C%22layout%22%3A%22xy-3d%22%2C%22position%22%3A%5B143944.0%2C61076.0%2C192.0%5D%7D) |
| 2 | SegB (`720575940615862389`) | [open](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B4e-09%2C%22m%22%5D%2C%22y%22%3A%5B4e-09%2C%22m%22%5D%2C%22z%22%3A%5B4e-08%2C%22m%22%5D%7D%2C%22crossSectionScale%22%3A2.0339912586467497%2C%22projectionOrientation%22%3A%5B-0.4705163836479187%2C0.8044001460075378%2C-0.30343097448349%2C0.1987067461013794%5D%2C%22projectionScale%22%3A13976.00585680798%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A//https%3A//bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22precomputed%3A//gs%3A//flywire_v141_m783%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22flywire_v141_m783%22%2C%22segments%22%3A%5B%22720575940615862389%22%5D%7D%5D%2C%22showDefaultAnnotations%22%3Afalse%2C%22selectedLayer%22%3A%7B%22size%22%3A350%2C%22layer%22%3A%22flywire_v141_m783%22%7D%2C%22layout%22%3A%22xy-3d%22%2C%22position%22%3A%5B145597.0%2C64135.0%2C108.0%5D%7D) |
| 3 | SegC (`720575940630914858`) | [open](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B4e-09%2C%22m%22%5D%2C%22y%22%3A%5B4e-09%2C%22m%22%5D%2C%22z%22%3A%5B4e-08%2C%22m%22%5D%7D%2C%22crossSectionScale%22%3A2.0339912586467497%2C%22projectionOrientation%22%3A%5B-0.4705163836479187%2C0.8044001460075378%2C-0.30343097448349%2C0.1987067461013794%5D%2C%22projectionScale%22%3A13976.00585680798%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A//https%3A//bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22precomputed%3A//gs%3A//flywire_v141_m783%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22flywire_v141_m783%22%2C%22segments%22%3A%5B%22720575940630914858%22%5D%7D%5D%2C%22showDefaultAnnotations%22%3Afalse%2C%22selectedLayer%22%3A%7B%22size%22%3A350%2C%22layer%22%3A%22flywire_v141_m783%22%7D%2C%22layout%22%3A%22xy-3d%22%2C%22position%22%3A%5B143828.0%2C61503.0%2C117.0%5D%7D) |
| 4 | SegD (`720575940623617973`) | [open](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B4e-09%2C%22m%22%5D%2C%22y%22%3A%5B4e-09%2C%22m%22%5D%2C%22z%22%3A%5B4e-08%2C%22m%22%5D%7D%2C%22crossSectionScale%22%3A2.0339912586467497%2C%22projectionOrientation%22%3A%5B-0.4705163836479187%2C0.8044001460075378%2C-0.30343097448349%2C0.1987067461013794%5D%2C%22projectionScale%22%3A13976.00585680798%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A//https%3A//bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22precomputed%3A//gs%3A//flywire_v141_m783%22%2C%22tab%22%3A%22source%22%2C%22name%22%3A%22flywire_v141_m783%22%2C%22segments%22%3A%5B%22720575940623617973%22%5D%7D%5D%2C%22showDefaultAnnotations%22%3Afalse%2C%22selectedLayer%22%3A%7B%22size%22%3A350%2C%22layer%22%3A%22flywire_v141_m783%22%7D%2C%22layout%22%3A%22xy-3d%22%2C%22position%22%3A%5B144623.0%2C59532.0%2C71.0%5D%7D) |

**Note:** JSON may round very large segment IDs in the URL fragment; if the wrong segment highlights, set the segment ID manually in Neuroglancer or generate links with segment IDs kept as strings in code.

---

## Setup

Requires a `.env` file with API keys:
```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

Dependencies: `selenium`, `Pillow`, `scipy`, `numpy`, `litellm`, `python-dotenv`.  
ChromeDriver is auto-detected via Selenium Manager.
