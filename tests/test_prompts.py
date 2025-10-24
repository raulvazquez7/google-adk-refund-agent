"""
Unit tests for prompt loading utilities.

Tests ensure prompts are loaded correctly and formatted properly.
"""
import pytest
from pathlib import Path

from src.utils.prompts import load_prompts, get_prompt


class TestPromptLoading:
    """Test prompt loading from config/prompts.yaml."""

    def test_load_prompts_returns_dict(self):
        """Test that load_prompts returns a dictionary."""
        prompts = load_prompts()
        assert isinstance(prompts, dict)
        assert len(prompts) > 0

    def test_load_prompts_cached(self):
        """Test that load_prompts uses caching."""
        # Call twice and verify it's the same object (cached)
        prompts1 = load_prompts()
        prompts2 = load_prompts()
        assert prompts1 is prompts2  # Same object reference

    def test_required_prompts_exist(self):
        """Test that required prompts exist in config."""
        prompts = load_prompts()
        required = ["intent_classification", "response_assembly"]
        for prompt_name in required:
            assert prompt_name in prompts, f"Missing required prompt: {prompt_name}"

    def test_prompts_are_strings(self):
        """Test that all prompts are strings."""
        prompts = load_prompts()
        for name, template in prompts.items():
            assert isinstance(template, str), f"Prompt '{name}' is not a string"


class TestGetPrompt:
    """Test get_prompt function."""

    def test_get_prompt_intent_classification(self):
        """Test getting intent classification prompt."""
        prompt = get_prompt("intent_classification", user_message="Can I return my order?")
        assert "Can I return my order?" in prompt
        assert "intent" in prompt.lower()
        assert "category" in prompt.lower()

    def test_get_prompt_response_assembly(self):
        """Test getting response assembly prompt."""
        prompt = get_prompt(
            "response_assembly",
            user_message="Test message",
            intent="refund",
            context_str="Test context",
            eligibility_context=""
        )
        assert "Test message" in prompt
        assert "refund" in prompt
        assert "Test context" in prompt

    def test_get_prompt_invalid_name(self):
        """Test that invalid prompt name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_prompt("nonexistent_prompt", user_message="test")

        assert "not found" in str(exc_info.value).lower()
        assert "nonexistent_prompt" in str(exc_info.value)

    def test_get_prompt_missing_variable(self):
        """Test that missing template variable raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            # Intent classification requires user_message
            get_prompt("intent_classification")

        assert "user_message" in str(exc_info.value).lower()

    def test_get_prompt_formatting(self):
        """Test that prompt formatting works correctly."""
        test_message = "I want a refund for ORD-12345"
        prompt = get_prompt("intent_classification", user_message=test_message)

        # Verify the message was inserted correctly
        assert test_message in prompt
        # Verify it's not just the raw template
        assert "{user_message}" not in prompt


class TestPromptTemplateStructure:
    """Test prompt template structure and content."""

    def test_intent_classification_has_categories(self):
        """Test that intent classification includes category definitions."""
        prompt = get_prompt("intent_classification", user_message="test")
        categories = ["refund", "policy", "general"]
        for category in categories:
            assert category in prompt.lower()

    def test_response_assembly_has_instructions(self):
        """Test that response assembly includes instructions."""
        prompt = get_prompt(
            "response_assembly",
            user_message="test",
            intent="refund",
            context_str="test",
            eligibility_context=""
        )
        keywords = ["friendly", "professional", "empathetic"]
        for keyword in keywords:
            assert keyword.lower() in prompt.lower()

    def test_response_assembly_has_response_types(self):
        """Test that response assembly includes all response types."""
        prompt = get_prompt(
            "response_assembly",
            user_message="test",
            intent="refund",
            context_str="test",
            eligibility_context=""
        )
        response_types = [
            "refund_eligible",
            "refund_not_eligible",
            "refund_already_processed",
            "policy_info",
            "general_info",
            "error"
        ]
        for response_type in response_types:
            assert response_type in prompt
