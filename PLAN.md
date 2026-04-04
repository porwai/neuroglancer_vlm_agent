# VLM Neuroglancer Navigation Benchmark

## Objective
Test whether current Vision-Language Models can navigate 3D brain electron microscopy data (FlyWire dataset) in Neuroglancer. Task: reach the highest Z position (deepest brain tissue) from a given neuron starting position.

---

## How It Works

Each step of the loop:
1. Take a screenshot of the Neuroglancer view + extract position/orientation state
2. Send image + state info to a VLM with a prompt asking "where should I move?"
3. VLM returns a JSON action (primarily `delta_z` to go deeper, plus orientation/zoom adjustments)
4. Feed action into `env.step()`, get new state + reward (Z-delta)
5. Repeat for 200 steps per episode

```
VLM receives: screenshot + state (position, orientation, zoom) + reward
VLM outputs:  JSON action {delta_x, delta_y, delta_z, orientation, zoom}
                          ↓
              env.step(action_vector)
                          ↓
              New state + reward (z_delta) → back to VLM
```

---

## Models to Compare

### VLM Agents
- **Claude Sonnet** (Anthropic API) — primary
- **GPT-4o** (OpenAI API) — comparison
- **Gemini Flash** (Google AI API) — fast/cheap baseline

### Non-VLM Baselines
- **Random agent** — random delta_z in [0, 50] each step
- **Greedy-Z agent** — always delta_z=100, everything else zero (tests if "just go up" works)
- **VLM no-image** — same prompt but no screenshot (ablation: does vision actually help?)

---

## Evaluation Design

### Starting Positions (16 total)

| # | Name | x | y | z | Category |
|---|------|---|---|---|----------|
| 1 | Low-Z central | 143944 | 61076 | 192 | Low-Z |
| 2 | Low-Z anterior | 120000 | 50000 | 100 | Low-Z |
| 3 | Low-Z posterior | 170000 | 70000 | 150 | Low-Z |
| 4 | Low-Z lateral-L | 100000 | 40000 | 200 | Low-Z |
| 5 | Low-Z lateral-R | 180000 | 80000 | 180 | Low-Z |
| 6 | Low-Z ventral | 150000 | 30000 | 120 | Low-Z |
| 7 | Low-Z dorsal | 155000 | 90000 | 160 | Low-Z |
| 8 | Mid-Z central | 143944 | 61076 | 1500 | Mid-Z |
| 9 | Mid-Z anterior | 125000 | 55000 | 2000 | Mid-Z |
| 10 | Mid-Z posterior | 165000 | 65000 | 1800 | Mid-Z |
| 11 | Mid-Z lateral | 110000 | 45000 | 2500 | Mid-Z |
| 12 | High-Z start | 143944 | 61076 | 3500 | High-Z |
| 13 | Near soma | 174232 | 68224 | 3870 | High-Z |
| 14 | Edge-X low | 50000 | 60000 | 300 | Edge |
| 15 | Edge-Y low | 140000 | 15000 | 250 | Edge |
| 16 | Random deep | 157510 | 42018 | 4661 | High-Z |

### Parameters
- **Steps per episode:** 200
- **Trials per configuration:** 3
- **Total episodes:** 16 positions x 6 agents x 3 trials = 288 episodes

### Timing
- Per step: ~2s (VLM call) + 50ms (env) + 500ms (render wait) ≈ 2.5s
- Per episode (200 steps): ~8 minutes
- VLM episodes only (16 x 3 models x 3 trials = 144): ~19 hours
- Baselines are near-instant (no API calls)

### Cost Estimate
- Claude Sonnet: ~$1.20/episode
- GPT-4o: ~$1.20/episode
- Gemini Flash: ~$0.12/episode
- **Total estimate: ~$150–200**

---

## Metrics

### Primary Metric
**Final Z reached** from each starting position — which model gets deepest?

### Per-Episode
- `start_z`, `final_z`, `max_z_reached`, `total_z_gain`
- `num_steps`, `total_time`, `estimated_cost`
- `num_negative_z_steps` (steps where Z decreased)
- `num_parse_failures` (VLM returned unparseable output)

### Aggregate (across all positions)
- Mean/std final Z per model
- **Win rate**: for each starting position, which model achieved highest Z
- **Efficiency**: Z gain per step, Z gain per dollar
- **Failure rate**: episodes where agent made no Z progress

---

## Reward Function

```python
def vlm_reward(state, action, prev_state):
    current_z = state[0][0][2]
    prev_z = prev_state[0][0][2]
    done = current_z >= 6500  # near max of 7062
    return current_z - prev_z, done
```

Reward = Z-delta each step. Communicated to VLM in the prompt as feedback.

---

## Prompt Design

### System Prompt (once per episode)
- Explain: you're navigating a 3D brain viewer, goal is maximize Z
- Define JSON action format with all fields
- Explain that deltas are applied directly to state (no internal scaling)
- Strategy hints: focus on delta_z, adjust orientation if view goes blank

### Per-Step Prompt
- Current state: position, orientation, scales
- Z progress: current vs start, cumulative gain
- Reward from last step
- Attached screenshot
- Request: output JSON action

### Output Parsing
- Regex extract JSON from VLM response
- Map to 16-element action vector (json_change=1, all clicks=0)
- Fallback on parse failure: default action (delta_z=5)

---

## Project Structure

```
vlm_navigator/
    config/
        vlm_config.json            # Model params, API key env var names
        starting_positions.json    # The 16 starting positions
    agents/
        base_agent.py              # Abstract base: get_action(), parse_response()
        claude_agent.py            # Claude Sonnet implementation
        gpt4o_agent.py             # GPT-4o implementation
        gemini_agent.py            # Gemini Flash implementation
        random_agent.py            # Random baseline
        greedy_agent.py            # Greedy-Z baseline
        prompt_templates.py        # System/step prompt templates
    evaluation/
        runner.py                  # Main eval loop (positions x models x trials)
        url_builder.py             # Build Neuroglancer URLs from position dicts
        metrics.py                 # Per-step logging, episode summaries
    analysis/
        compare_models.py          # Aggregate stats, comparison tables
        plot_results.py            # Visualizations
    utils/
        image_utils.py             # Screenshot resize, base64 encode, blank detect
        action_utils.py            # VLM JSON -> action vector conversion
    results/                       # Generated at runtime (gitignored)
    run_evaluation.py              # Entry point
    run_analysis.py                # Generate comparison report
    requirements.txt               # anthropic, openai, google-generativeai, etc.
```

---

## Implementation Phases

### Phase 1 — Foundation
- `action_utils.py`, `url_builder.py`, `starting_positions.json`, `base_agent.py`

### Phase 2 — Single Agent + Prompt Tuning
- Implement `claude_agent.py` with prompt templates
- Manual test: 1 position, 20 steps, observe and iterate on prompt

### Phase 3 — Metrics + Runner
- Implement `metrics.py` and `runner.py`
- Run a full 200-step episode, verify logging

### Phase 4 — Additional Agents
- Implement `gpt4o_agent.py`, `gemini_agent.py`, `random_agent.py`, `greedy_agent.py`

### Phase 5 — Full Evaluation
- Run all episodes (can parallelize: 1 Chrome instance per model)

### Phase 6 — Analysis
- Run comparison scripts, generate plots and tables

---

## Key Technical Notes

- `apply_actions()` in ngllib applies deltas **directly** to JSON state (no scaling multipliers)
- Add `time.sleep(0.5)` after `env.step()` to let Neuroglancer render before screenshotting
- Resize screenshots to 960x540 before sending to VLM (saves tokens/cost)
- FlyWire Z range: [16, 7062]
- Neuroglancer URL change = full page reload — may need render wait
- Parse failures need graceful fallback (don't crash the episode)

## Key Question
The interesting question isn't just "can the VLM increase Z" (a greedy baseline can do that). It's whether the VLM can **use visual information** to stay on neural tissue, recover when it loses structure, and navigate more intelligently than a blind agent.
