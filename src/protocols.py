"""
Communication protocols for agent-to-agent (A2A) interactions.

This module defines the standard message format that all agents use to
communicate with each other. It ensures consistency, traceability, and
error handling across the multi-agent system.
"""
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class AgentRequest(BaseModel):
    """
    Standard request format for agent-to-agent communication.
    
    This is sent by the Coordinator to specialized agents when delegating tasks.
    
    Attributes:
        agent: Target agent name (e.g., "policy_expert", "transaction_agent")
        task: Specific task to perform (e.g., "check_policy", "get_order")
        context: Task-specific data needed by the agent
        metadata: Tracing and session information
    """
    agent: str = Field(..., description="Target agent identifier")
    task: str = Field(..., description="Task to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context and parameters")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session, user, and trace information"
    )


class AgentResponse(BaseModel):
    """
    Standard response format from agents.
    
    Every agent returns this format, ensuring consistent error handling
    and result processing.
    
    Attributes:
        agent: Agent that generated the response
        status: Success or error indicator
        result: Task output (if successful)
        error: Error details (if failed)
        metadata: Performance and debugging information
    """
    agent: str = Field(..., description="Agent that generated this response")
    status: Literal["success", "error"] = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": datetime.now().isoformat(),
            "latency_ms": 0,
            "tokens_used": 0
        },
        description="Performance metrics and debugging info"
    )
    
    @classmethod
    def create_success(cls, agent: str, result: Dict[str, Any], **metadata_kwargs) -> "AgentResponse":
        """Helper to create a successful response."""
        return cls(
            agent=agent,
            status="success",
            result=result,
            metadata={
                "timestamp": datetime.now().isoformat(),
                **metadata_kwargs
            }
        )
    
    @classmethod
    def create_error(cls, agent: str, error_message: str, **metadata_kwargs) -> "AgentResponse":
        """Helper to create an error response."""
        return cls(
            agent=agent,
            status="error",
            error=error_message,
            metadata={
                "timestamp": datetime.now().isoformat(),
                **metadata_kwargs
            }
        )
