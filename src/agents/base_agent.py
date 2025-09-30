"""
Base class for all specialized agents in the multi-agent system.

All agents inherit from this class to ensure consistent behavior,
error handling, and observability across the system.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any

from langfuse import Langfuse
from src.protocols import AgentRequest, AgentResponse


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality:
    - Request/response handling
    - Error management
    - Performance tracking
    - Observability integration
    
    Subclasses must implement the `_execute_task()` method with their
    specific logic.
    """
    
    def __init__(self, name: str, tracer: Langfuse):
        """
        Initialize the agent.
        
        Args:
            name: Agent identifier (e.g., "policy_expert")
            tracer: Langfuse client for observability
        """
        self.name = name
        self.tracer = tracer
        logger.info(f"Initialized agent: {self.name}")
    
    async def handle_request(self, request: AgentRequest) -> AgentResponse:
        """
        Main entry point for handling agent requests.
        
        This method:
        1. Validates the request
        2. Starts tracing
        3. Executes the task
        4. Handles errors
        5. Returns standardized response
        
        Args:
            request: Validated agent request
            
        Returns:
            Standardized agent response
        """
        start_time = time.time()
        
        # Start tracing span for this agent
        with self.tracer.start_as_current_span(name=f"{self.name}_{request.task}"):
            self.tracer.update_current_trace(
                input=request.model_dump(),
                tags=[self.name, request.task]
            )
            
            try:
                logger.info(f"[{self.name}] Executing task: {request.task}")
                
                # Call the subclass-specific implementation
                result = await self._execute_task(request)
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                response = AgentResponse.create_success(
                    agent=self.name,
                    result=result,
                    latency_ms=latency_ms
                )
                
                self.tracer.update_current_trace(output=response.model_dump())
                logger.info(f"[{self.name}] Task completed successfully in {latency_ms}ms")
                
                return response
                
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                error_msg = f"Error in {self.name}: {str(e)}"
                
                logger.error(error_msg, exc_info=True)
                
                response = AgentResponse.create_error(
                    agent=self.name,
                    error_message=error_msg,
                    latency_ms=latency_ms
                )
                
                # Update trace with error info (using tags instead of level)
                self.tracer.update_current_trace(
                    output=response.model_dump(),
                    tags=[self.name, request.task, "error"]
                )
                
                return response
    
    @abstractmethod
    async def _execute_task(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Execute the specific task logic for this agent.
        
        This method MUST be implemented by all subclasses.
        
        Args:
            request: The validated agent request
            
        Returns:
            Task result as a dictionary
            
        Raises:
            NotImplementedError: If subclass doesn't implement this
            Exception: Any task-specific errors
        """
        raise NotImplementedError(f"Agent {self.name} must implement _execute_task()")
