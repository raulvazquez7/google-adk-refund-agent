"""
Pydantic models for data validation and structured outputs.

This module centralizes all data contracts used across the application:
- API models (request/response schemas)
- LLM structured outputs (response templates)
- Agent protocols (A2A communication)
"""
from src.models.schemas import (
    OrderItem,
    OrderData,
    OrderResponse,
    RefundResponse,
    RefundEligibilityInfo,
    AgentResponseTemplate,
    IntentClassification
)
from src.models.protocols import AgentRequest, AgentResponse

__all__ = [
    # LLM Structured Outputs
    "RefundResponse",
    "RefundEligibilityInfo",
    "AgentResponseTemplate",
    "IntentClassification",

    # Tool Schemas
    "OrderItem",
    "OrderData",
    "OrderResponse",

    # Agent Protocols
    "AgentRequest",
    "AgentResponse",
]
