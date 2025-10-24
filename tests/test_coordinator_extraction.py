"""
Unit tests for coordinator's order ID extraction logic.
"""
import pytest
from langfuse import Langfuse

from src.agents.coordinator import CoordinatorAgent
from src.agents.policy_expert import PolicyExpertAgent
from src.agents.transaction_agent import TransactionAgent


class TestOrderIDExtraction:
    """Test order ID extraction from various user inputs."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator instance for testing."""
        langfuse = Langfuse()
        policy_expert = PolicyExpertAgent(tracer=langfuse)
        transaction_agent = TransactionAgent(tracer=langfuse)

        return CoordinatorAgent(
            tracer=langfuse,
            specialized_agents={
                "policy_expert": policy_expert,
                "transaction_agent": transaction_agent
            }
        )

    def test_extract_full_format_uppercase(self, coordinator):
        """Test extraction with full ORD-XXXXX format (uppercase)."""
        text = "I want to return my order ORD-84315"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-84315"

    def test_extract_full_format_lowercase(self, coordinator):
        """Test extraction with full ord-xxxxx format (lowercase)."""
        text = "I want to return my order ord-84315"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-84315"

    def test_extract_number_only_with_is(self, coordinator):
        """Test extraction from 'order is 44012' format."""
        text = "Thank you, now I want to return other purchase, the order is 44012"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-44012"

    def test_extract_number_only_direct(self, coordinator):
        """Test extraction from 'order 44012' format."""
        text = "I want to return order 44012"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-44012"

    def test_extract_number_with_number_keyword(self, coordinator):
        """Test extraction from 'order number 123456' format."""
        text = "Can I return order number 123456?"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-123456"

    def test_extract_number_with_hash(self, coordinator):
        """Test extraction from 'order #789' format."""
        text = "I need to return order #159753"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-159753"

    def test_no_order_id_in_text(self, coordinator):
        """Test that None is returned when no order ID is present."""
        text = "What is your refund policy?"
        result = coordinator._extract_order_id(text)
        assert result is None

    def test_short_number_not_extracted(self, coordinator):
        """Test that short numbers (< 4 digits) are not extracted."""
        text = "I want to return order 123"
        result = coordinator._extract_order_id(text)
        assert result is None

    def test_multiple_order_ids_first_one_wins(self, coordinator):
        """Test that when multiple IDs exist, the first one is extracted."""
        text = "I want to return ORD-84315 and ORD-44012"
        result = coordinator._extract_order_id(text)
        assert result == "ORD-84315"

    def test_prefer_full_format_over_number(self, coordinator):
        """Test that full format (ORD-X) is preferred over number format."""
        text = "I want to return order 44012 which is ORD-159753"
        result = coordinator._extract_order_id(text)
        # Should match ORD-159753 first
        assert result == "ORD-159753"
