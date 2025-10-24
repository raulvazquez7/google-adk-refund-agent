"""
Unit tests for Pydantic schemas validation.

Tests ensure that structured outputs work correctly and validate data properly.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.schemas import (
    IntentClassification,
    AgentResponseTemplate,
    RefundEligibilityInfo,
    OrderData,
    OrderResponse,
    RefundProcessingResult,
    OrderItem
)


class TestIntentClassification:
    """Test IntentClassification schema."""

    def test_valid_intent_classification(self):
        """Test valid intent classification."""
        data = {
            "intent": "refund",
            "confidence": 0.95
        }
        classification = IntentClassification(**data)
        assert classification.intent == "refund"
        assert classification.confidence == 0.95

    def test_invalid_intent_type(self):
        """Test that invalid intent types are rejected."""
        data = {
            "intent": "invalid_type",
            "confidence": 0.8
        }
        with pytest.raises(ValidationError):
            IntentClassification(**data)

    def test_confidence_out_of_range(self):
        """Test that confidence must be between 0.0 and 1.0."""
        data = {
            "intent": "policy",
            "confidence": 1.5
        }
        with pytest.raises(ValidationError):
            IntentClassification(**data)


class TestAgentResponseTemplate:
    """Test AgentResponseTemplate schema."""

    def test_valid_response_template(self):
        """Test valid response template."""
        data = {
            "response_type": "refund_eligible",
            "message": "Your order qualifies for a refund.",
            "action_required": "Reply 'yes' to confirm",
            "key_details": ["Order is 5 days old", "Delivered status", "Within 14-day window"]
        }
        template = AgentResponseTemplate(**data)
        assert template.response_type == "refund_eligible"
        assert len(template.key_details) == 3

    def test_invalid_response_type(self):
        """Test that invalid response_type fails validation."""
        data = {
            "response_type": "invalid_type",
            "message": "Test"
        }
        with pytest.raises(ValidationError):
            AgentResponseTemplate(**data)

    def test_message_max_length(self):
        """Test that message max length is enforced."""
        data = {
            "response_type": "general_info",
            "message": "A" * 1001  # Exceeds 1000 char limit
        }
        with pytest.raises(ValidationError):
            AgentResponseTemplate(**data)

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        data = {
            "response_type": "policy_info",
            "message": "Here's our policy."
        }
        template = AgentResponseTemplate(**data)
        assert template.action_required == ""
        assert template.key_details == []


class TestRefundEligibilityInfo:
    """Test RefundEligibilityInfo schema."""

    def test_eligible_order(self):
        """Test eligible order info."""
        data = {
            "eligible": True,
            "reason": "Order is within 14-day window",
            "order_status": "DELIVERED",
            "days_since_purchase": 5,
            "days_remaining": 9
        }
        info = RefundEligibilityInfo(**data)
        assert info.eligible is True
        assert info.days_since_purchase == 5

    def test_already_refunded(self):
        """Test already refunded order."""
        data = {
            "eligible": False,
            "reason": "Order was already refunded",
            "order_status": "RETURNED",
            "already_refunded": True,
            "refund_transaction_id": "REF-12345",
            "refund_date": "2025-01-15T10:00:00Z",
            "refund_amount": 89.99
        }
        info = RefundEligibilityInfo(**data)
        assert info.already_refunded is True
        assert info.refund_transaction_id == "REF-12345"

    def test_invalid_days_negative(self):
        """Test that negative days are rejected."""
        data = {
            "eligible": False,
            "reason": "Test",
            "order_status": "DELIVERED",
            "days_since_purchase": -5
        }
        with pytest.raises(ValidationError):
            RefundEligibilityInfo(**data)


class TestOrderSchemas:
    """Test order-related schemas."""

    def test_valid_order_item(self):
        """Test valid order item."""
        data = {
            "name": "Classic Barefoot Sneaker",
            "price": 89.99
        }
        item = OrderItem(**data)
        assert item.name == "Classic Barefoot Sneaker"
        assert item.price == 89.99

    def test_order_item_negative_price_validation(self):
        """Test that price cannot be negative."""
        data = {
            "name": "Test Shoe",
            "price": -50.0
        }
        with pytest.raises(ValidationError):
            OrderItem(**data)

    def test_valid_order_data(self):
        """Test valid order data."""
        data = {
            "order_id": "ORD-84315",
            "user_id": "user-gkw",
            "purchase_date": datetime.now().isoformat(),
            "status": "DELIVERED",
            "items": [
                {
                    "name": "Classic Sneaker",
                    "price": 89.99
                }
            ]
        }
        order = OrderData(**data)
        assert order.order_id == "ORD-84315"
        assert order.user_id == "user-gkw"
        assert len(order.items) == 1

    def test_invalid_order_id_pattern(self):
        """Test that order_id must match pattern."""
        data = {
            "order_id": "INVALID-ID",
            "user_id": "user-123",
            "purchase_date": datetime.now().isoformat(),
            "status": "DELIVERED",
            "items": []
        }
        with pytest.raises(ValidationError):
            OrderData(**data)

    def test_order_response_found(self):
        """Test OrderResponse with found order."""
        data = {
            "found": True,
            "order_data": {
                "order_id": "ORD-12345",
                "user_id": "user-test",
                "purchase_date": datetime.now().isoformat(),
                "status": "DELIVERED",
                "items": []
            }
        }
        response = OrderResponse(**data)
        assert response.found is True
        assert response.order_data is not None

    def test_order_response_not_found(self):
        """Test OrderResponse with order not found."""
        data = {
            "found": False,
            "error": "Order ORD-99999 not found"
        }
        response = OrderResponse(**data)
        assert response.found is False
        assert response.order_data is None


class TestRefundProcessingResult:
    """Test RefundProcessingResult schema."""

    def test_successful_refund(self):
        """Test successful refund result."""
        data = {
            "success": True,
            "order_id": "ORD-12345",
            "transaction_id": "REF-67890",
            "amount": 89.99,
            "refund_date": "2025-01-15T10:00:00Z"
        }
        result = RefundProcessingResult(**data)
        assert result.success is True
        assert result.transaction_id == "REF-67890"

    def test_failed_refund(self):
        """Test failed refund result."""
        data = {
            "success": False,
            "order_id": "ORD-12345",
            "error": "Order already refunded"
        }
        result = RefundProcessingResult(**data)
        assert result.success is False
        assert result.error == "Order already refunded"

    def test_negative_amount_rejected(self):
        """Test that negative amounts are rejected."""
        data = {
            "success": True,
            "order_id": "ORD-12345",
            "transaction_id": "REF-12345",
            "amount": -50.0,
            "refund_date": "2025-01-15T10:00:00Z"
        }
        with pytest.raises(ValidationError):
            RefundProcessingResult(**data)
