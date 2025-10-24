"""
Prompt loading utilities with caching.

This module provides efficient loading of prompt templates from config/prompts.yaml.
Prompts are cached to avoid repeated file reads.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict
import yaml


@lru_cache(maxsize=1)
def load_prompts() -> Dict[str, str]:
    """
    Load prompts from config/prompts.yaml.

    Uses LRU cache to avoid repeated file reads (loaded once per process).

    Returns:
        Dict mapping prompt names to template strings

    Raises:
        FileNotFoundError: If prompts.yaml doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Prompts configuration not found at {config_path}. "
            "Ensure config/prompts.yaml exists."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_prompt(name: str, **kwargs) -> str:
    """
    Get prompt by name and format with kwargs.

    Args:
        name: Prompt name (e.g., "intent_classification", "response_assembly")
        **kwargs: Variables to format into prompt template

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If prompt name not found
        KeyError: If required template variable is missing

    Example:
        >>> prompt = get_prompt("intent_classification", user_message="Can I return my order?")
        >>> print(prompt)
        Classify the user's intent...
    """
    prompts = load_prompts()
    template = prompts.get(name)

    if not template:
        available = list(prompts.keys())
        raise ValueError(
            f"Prompt '{name}' not found in config/prompts.yaml. "
            f"Available prompts: {available}"
        )

    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Missing required variable {e} for prompt '{name}'. "
            f"Provided: {list(kwargs.keys())}"
        ) from e
