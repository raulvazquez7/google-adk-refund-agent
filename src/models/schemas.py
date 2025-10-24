"""
Pydantic schemas for structured outputs and data validation.

This module defines:
1. LLM structured outputs (response templates)
2. Tool data schemas (orders, refunds, etc.)
3. Business logic models (eligibility, etc.)
"""
from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field


# ============================================================================
# LLM STRUCTURED OUTPUTS (Response Templates)
# ============================================================================

class IntentClassification(BaseModel):
    """
    Structured output for intent classification.

    Forces LLM to return a valid intent category.
    """
    intent: Literal["refund", "policy", "general"] = Field(
        ...,
        description="User's intent category"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for classification"
    )


class RefundEligibilityInfo(BaseModel):
    """
    Structured refund eligibility information.

    This replaces the giant dict returned by _check_refund_eligibility().
    """
    eligible: bool = Field(..., description="Whether order qualifies for refund")
    reason: str = Field(..., description="Explanation of eligibility decision")

    # Status flags
    already_refunded: bool = Field(
        default=False,
        description="Whether order was already refunded"
    )
    invalid_status: bool = Field(
        default=False,
        description="Whether order status prevents refund"
    )

    # Order details
    order_status: str = Field(..., description="Current order status")
    days_since_purchase: Optional[int] = Field(
        None,
        ge=0,
        description="Days elapsed since purchase"
    )
    days_remaining: Optional[int] = Field(
        None,
        ge=0,
        description="Days remaining in refund window (if eligible)"
    )

    # Refund details (if already refunded)
    refund_transaction_id: Optional[str] = Field(
        None,
        description="Transaction ID if already refunded"
    )
    refund_date: Optional[str] = Field(
        None,
        description="Refund date if already refunded"
    )
    refund_amount: Optional[float] = Field(
        None,
        ge=0,
        description="Refund amount if already refunded"
    )


class AgentResponseTemplate(BaseModel):
    """
    Structured template for final user-facing responses.

    This replaces the string-based prompt assembly in coordinator.py.
    LLM will fill this template with appropriate values.
    """
    response_type: Literal[
        "refund_eligible",
        "refund_not_eligible",
        "refund_already_processed",
        "policy_info",
        "general_info",
        "error"
    ] = Field(..., description="Type of response being provided")

    message: str = Field(
        ...,
        max_length=1000,
        description="Main response message to the user (friendly, empathetic, concise)"
    )

    action_required: str = Field(
        default="",
        description="What user should do next (e.g., 'Reply yes to confirm refund'). Leave empty if no action needed."
    )

    key_details: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Bullet points of important information (max 5). Leave empty list if none."
    )


class RefundResponse(BaseModel):
    """
    Complete response for refund-related queries.

    Combines eligibility info with the templated response.
    """
    eligibility: Optional[RefundEligibilityInfo] = Field(
        None,
        description="Eligibility information if order was checked"
    )

    response: AgentResponseTemplate = Field(
        ...,
        description="User-facing response"
    )


# ============================================================================
# TOOL DATA SCHEMAS (Orders, Items, etc.)
# ============================================================================

class OrderItem(BaseModel):
    """Individual item in an order."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., ge=0, description="Item price in USD")


class OrderData(BaseModel):
    """
    Complete order information from Firestore.

    Replaces the untyped dict currently returned by get_order_details.
    """
    order_id: str = Field(
        ...,
        pattern=r"^ORD-\d+$",
        description="Order identifier"
    )
    user_id: str = Field(..., description="User identifier")
    purchase_date: datetime = Field(..., description="Purchase timestamp")
    status: Literal["PENDING", "SHIPPED", "DELIVERED", "RETURNED", "CANCELLED"] = Field(
        ...,
        description="Current order status"
    )
    items: List[OrderItem] = Field(..., description="Ordered items")

    # Optional refund fields
    refund_transaction_id: Optional[str] = Field(
        None,
        description="Refund transaction ID if returned"
    )
    refund_date: Optional[str] = Field(
        None,
        description="Refund processing date if returned"
    )
    refund_amount: Optional[float] = Field(
        None,
        ge=0,
        description="Refund amount if returned"
    )


class OrderResponse(BaseModel):
    """
    Response from get_order_details tool.

    Standardizes tool output format.
    """
    found: bool = Field(..., description="Whether order was found in database")
    order_data: Optional[OrderData] = Field(
        None,
        description="Order details if found"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if not found"
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "found": True,
                "order_data": {
                    "order_id": "ORD-84315",
                    "user_id": "user-gkw",
                    "purchase_date": "2025-09-18T10:00:00Z",
                    "status": "DELIVERED",
                    "items": [
                        {
                            "name": "Sandalias Zénit Play",
                            "price": 45.50
                        }
                    ]
                },
                "error": None
            }
        ]
    }}


class RefundProcessingResult(BaseModel):
    """
    Result from process_refund tool.

    Standardizes refund processing output.
    """
    success: bool = Field(..., description="Whether refund was processed successfully")
    order_id: str = Field(..., description="Order ID that was refunded")
    transaction_id: Optional[str] = Field(
        None,
        description="Refund transaction ID if successful"
    )
    amount: Optional[float] = Field(
        None,
        ge=0,
        description="Refund amount if successful"
    )
    refund_date: Optional[str] = Field(
        None,
        description="Refund processing timestamp if successful"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
