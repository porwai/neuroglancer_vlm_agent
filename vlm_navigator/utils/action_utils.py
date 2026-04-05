"""
VLM response parsing and action vector conversion.

Converts VLM JSON output → 17-element action vector for ngllib Environment.step().

Action vector (Euler mode, 17 elements):
  [left_click, right_click, double_click,   # 3 booleans
   x, y,                                    # 2 floats (mouse)
   key_Shift, key_Ctrl, key_Alt,            # 3 booleans
   json_change,                             # 1 boolean
   delta_x, delta_y, delta_z,              # 3 floats (position)
   delta_crossSectionScale,                 # 1 float
   delta_e1, delta_e2, delta_e3,           # 3 floats (orientation)
   delta_projectionScale]                   # 1 float
"""

import json
import re

FALLBACK_ACTION = {"delta_x": 0, "delta_y": 0, "delta_z": 5}


def parse_vlm_response(raw_text: str) -> dict:
    """Extract a JSON action dict from VLM text output.

    Handles:
      - Pure JSON string
      - JSON inside markdown code blocks (```json ... ``` or ``` ... ```)
      - JSON embedded in surrounding text

    Returns FALLBACK_ACTION on any parse failure.
    """
    # Try 1: direct parse
    try:
        return json.loads(raw_text.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # Try 2: extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try 3: find first { ... } in the text
    match = re.search(r"\{[^{}]*\}", raw_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return FALLBACK_ACTION.copy()


def vlm_json_to_action_vector(vlm_output: dict, mode: str = "position_only") -> list:
    """Convert parsed VLM JSON to a 17-element action vector.

    Args:
        vlm_output: Dict with delta keys (delta_x, delta_y, delta_z, etc.)
        mode: "position_only" (Mode B) or "full" (Mode A)

    Returns:
        17-element list for Environment.step() in Euler mode.
    """
    dx = vlm_output.get("delta_x", 0)
    dy = vlm_output.get("delta_y", 0)
    dz = vlm_output.get("delta_z", 0)

    if mode == "full":
        de1 = vlm_output.get("delta_e1", 0)
        de2 = vlm_output.get("delta_e2", 0)
        de3 = vlm_output.get("delta_e3", 0)
        dcross = vlm_output.get("delta_cross", 0)
        dproj = vlm_output.get("delta_proj", 0)
    else:
        de1 = de2 = de3 = dcross = dproj = 0

    return [
        0, 0, 0,           # no clicks
        0, 0,              # no mouse position
        0, 0, 0,           # no modifier keys
        1,                 # json_change = True
        dx, dy, dz,        # position deltas
        dcross,            # crossSectionScale delta
        de1, de2, de3,     # orientation deltas (Euler)
        dproj,             # projectionScale delta
    ]
