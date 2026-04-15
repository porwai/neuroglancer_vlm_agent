"""
VLM Agent for Neuroglancer navigation.

Uses LiteLLM to call any vision model (Claude, GPT-4o, Gemini, etc.)
with a screenshot + state each step, parses the JSON response into
an action vector for the environment.

Prompt templates are kept separate — import and pass them in via config
or override get_system_prompt() / get_step_prompt().
"""

import base64
import io
import json

from litellm import completion
from PIL import Image

from vlm_navigator.utils.action_utils import parse_vlm_response, vlm_json_to_action_vector


class VLMAgent:
    def __init__(self, model_config: dict, action_mode: str = "position_only"):
        """
        Args:
            model_config: Dict with keys:
                - "model": LiteLLM model string (e.g. "anthropic/claude-sonnet-4-20250514")
                - "max_tokens": Max tokens for VLM response (default 300)
                - "temperature": Sampling temperature (default 0.0)
            action_mode: "position_only" (Mode B) or "full" (Mode A)
        """
        self.model = model_config["model"]
        self.max_tokens = model_config.get("max_tokens", 300)
        self.temperature = model_config.get("temperature", 0.0)
        self.action_mode = action_mode

        # Conversation history — system message + prior turns
        self.messages = []
        self.step_count = 0
        self.parse_failures = 0

    def reset(self, system_prompt: str):
        """Reset agent state for a new episode.

        Args:
            system_prompt: The system message that sets task context.
        """
        self.messages = [{"role": "system", "content": system_prompt}]
        self.step_count = 0
        self.parse_failures = 0

    def get_action(
        self,
        screenshot: Image.Image,
        position: list,
        orientation: list,
        cross_section_scale: float,
        projection_scale: float,
        prev_z_delta: float = 0.0,
        step_prompt_template: str = None,
    ) -> tuple[list, dict, str]:
        """Run one step: send screenshot + state to VLM, return action vector.

        Args:
            screenshot: PIL Image of current Neuroglancer view.
            position: [x, y, z] current position.
            orientation: [e1, e2, e3] current Euler orientation.
            cross_section_scale: Current crossSectionScale.
            projection_scale: Current projectionScale.
            prev_z_delta: Z change from the previous step (for feedback).
            step_prompt_template: Format string for the per-step user message.
                Should contain placeholders: {position}, {orientation},
                {cross_section_scale}, {projection_scale}, {step},
                {prev_z_delta}, {current_z}.
                If None, uses a minimal default.

        Returns:
            (action_vector, parsed_json, raw_response)
        """
        self.step_count += 1

        # Build the text portion of the user message.
        # step_prompt_template is expected to be already fully formatted by the caller.
        if step_prompt_template:
            step_text = step_prompt_template
        else:
            step_text = (
                f"Step {self.step_count}. "
                f"Position: {position}, Orientation: {orientation}, "
                f"CrossSectionScale: {cross_section_scale:.4f}, "
                f"ProjectionScale: {projection_scale:.2f}. "
                f"Last Z delta: {prev_z_delta:.1f}. "
                f"Current Z: {position[2]:.1f}. "
                f"Respond with a JSON action."
            )

        # Encode screenshot as base64 JPEG
        image_b64 = self._encode_image(screenshot)

        # Build user message with image + text
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": step_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
        self.messages.append(user_message)

        # Call VLM
        # Reasoning models (gpt-5, o1, etc.) use max_completion_tokens instead of max_tokens
        is_reasoning = any(tag in self.model for tag in ["gpt-5", "o1", "o3"])
        call_kwargs = dict(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            timeout=60.0,
        )
        if is_reasoning:
            call_kwargs["max_completion_tokens"] = self.max_tokens
        else:
            call_kwargs["max_tokens"] = self.max_tokens

        response = completion(**call_kwargs)
        msg = response.choices[0].message
        raw_text = msg.content or getattr(msg, "reasoning_content", None) or ""

        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": raw_text})

        # Parse response → action vector
        parsed = parse_vlm_response(raw_text)
        if parsed == {"delta_x": 0, "delta_y": 0, "delta_z": 5}:
            self.parse_failures += 1

        action_vector = vlm_json_to_action_vector(parsed, mode=self.action_mode)

        return action_vector, parsed, raw_text

    def trim_history(self, keep_last_n: int = 10):
        """Trim conversation history to avoid context overflow.

        Keeps the system message + the last N user/assistant turn pairs.
        """
        if len(self.messages) <= 1:
            return
        # messages[0] is system, then pairs of (user, assistant)
        non_system = self.messages[1:]
        if len(non_system) > keep_last_n * 2:
            self.messages = [self.messages[0]] + non_system[-(keep_last_n * 2):]

    @staticmethod
    def _encode_image(image: Image.Image, max_size: tuple = (960, 540)) -> str:
        """Resize and encode a PIL Image as base64 JPEG string."""
        if image.size != max_size:
            image = image.resize(max_size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
