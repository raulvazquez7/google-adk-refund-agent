"""
Test agent for validating the base agent infrastructure.
This will be deleted once real agents are implemented.
"""
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.protocols import AgentRequest


class TestAgent(BaseAgent):
    """Simple test agent that echoes back its input."""
    
    async def _execute_task(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Echo the request context back as result.
        
        This validates that:
        - Request/response flow works
        - Tracing is operational
        - Error handling functions
        """
        # Simulate some processing
        import asyncio
        await asyncio.sleep(0.5)
        
        return {
            "message": f"TestAgent received task '{request.task}'",
            "context_received": request.context,
            "agent_status": "operational"
        }
