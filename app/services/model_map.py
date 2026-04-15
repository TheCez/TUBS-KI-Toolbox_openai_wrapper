"""
Shared model mapping utilities.
Maps Anthropic model identifiers to TU-BS backend models, with environment variable override support.
"""

import os
import json
from typing import Optional
from app.models.tubs import CloudModel, LocalModel


def get_anthropic_model_map() -> dict:
    """
    Returns the Anthropic-to-TUBS model mapping dictionary.
    Checks ANTHROPIC_MODEL_MAP env var first, falls back to sensible defaults.
    """
    env_map = os.getenv("ANTHROPIC_MODEL_MAP")
    if env_map:
        try:
            return json.loads(env_map)
        except (json.JSONDecodeError, TypeError):
            pass

    # Default mapping: Anthropic model -> TU-BS equivalent
    return {
        "claude-opus-4-1": CloudModel.GPT_5_4.value,
        "claude-opus-4-1-20250805": CloudModel.GPT_5_4.value,
        "claude-opus-4-0": CloudModel.GPT_5_4.value,
        "claude-opus-4-20250514": CloudModel.GPT_5_4.value,
        "claude-sonnet-4-0": CloudModel.GPT_5_2.value,
        "claude-sonnet-4-20250514": CloudModel.GPT_5_2.value,
        "claude-3-5-opus-latest": CloudModel.GPT_5_4.value,
        "claude-3-opus-20240229": CloudModel.GPT_5_4.value,
        "claude-3-5-sonnet-latest": CloudModel.GPT_O3.value,
        "claude-3-5-sonnet-20241022": CloudModel.GPT_4O.value,
        "claude-3-5-sonnet-20240620": CloudModel.GPT_4O.value,
        "claude-3-7-sonnet-20250219": CloudModel.GPT_O3.value,
        "claude-3-7-sonnet-latest": CloudModel.GPT_O3.value,
        "claude-3-5-haiku-latest": CloudModel.GPT_O4_MINI.value,
        "claude-3-5-haiku-20241022": CloudModel.GPT_O4_MINI.value,
        "claude-3-haiku-20240307": CloudModel.GPT_4O_MINI.value,
    }


def resolve_model(model_id: str) -> str:
    """
    Resolves a model identifier to a TU-BS model name.
    If the model_id is an Anthropic alias, it's translated.
    If it's already a native TU-BS model, it passes through unchanged.
    """
    # Handle Enum values
    if hasattr(model_id, "value"):
        model_id = model_id.value

    model_str = str(model_id)
    anthropic_map = get_anthropic_model_map()
    return anthropic_map.get(model_str, model_str)
