"""
Base class for all specialized agents in the multi-agent system.

All agents inherit from this class to ensure consistent behavior,
error handling, and observability across the system.
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from langfuse import Langfuse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.config import settings
from src.models.protocols import AgentRequest, AgentResponse
from src.utils.logger import get_logger
from src.utils.rate_limiters import RateLimiters


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common functionality:
    - Request/response handling
    - Error management
    - Performance tracking
    - Observability integration
    - LLM timeout, rate limiting, and retry logic

    Subclasses must implement the `_execute_task()` method with their
    specific logic.

    Example:
        class PolicyExpert(BaseAgent):
            async def _execute_task(self, request: AgentRequest) -> Any:
                # Your agent logic here
                return {"policy": "refund within 14 days"}
    """

    def __init__(self, name: str, tracer: Langfuse):
        """
        Initialize the agent.

        Args:
            name: Agent identifier (e.g., "policy_expert", "transaction_agent")
            tracer: Langfuse client for observability
        """
        self.name = name
        self.tracer = tracer
        self.logger = get_logger(f"agent.{name}")

        self.logger.info(
            "agent_initialized",
            agent_name=name,
            llm_timeout=settings.llm_timeout,
            llm_max_retries=settings.llm_max_retries,
            llm_rate_limit=settings.llm_rate_limit
        )

    async def handle_request(self, request: AgentRequest) -> AgentResponse:
        """
        Main entry point for handling agent requests.

        This method:
        1. Validates the request
        2. Starts tracing span
        3. Executes the task
        4. Handles errors gracefully
        5. Returns standardized response

        Args:
            request: Validated agent request

        Returns:
            Standardized agent response (success or error)
        """
        start_time = time.time()

        # Start tracing span for this agent task
        with self.tracer.start_as_current_span(name=f"{self.name}_{request.task}"):
            self.tracer.update_current_trace(
                input=request.model_dump(),
                tags=[self.name, request.task]
            )

            try:
                self.logger.info(
                    "agent_task_started",
                    agent=self.name,
                    task=request.task,
                    context_keys=list(request.context.keys())
                )

                # Call the subclass-specific implementation
                result = await self._execute_task(request)

                latency_ms = int((time.time() - start_time) * 1000)

                response = AgentResponse.create_success(
                    agent=self.name,
                    result=result,
                    latency_ms=latency_ms
                )

                self.tracer.update_current_trace(output=response.model_dump())

                self.logger.info(
                    "agent_task_completed",
                    agent=self.name,
                    task=request.task,
                    latency_ms=latency_ms
                )

                return response

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                error_msg = f"Task failed in {self.name}: {str(e)}"

                self.logger.error(
                    "agent_task_failed",
                    agent=self.name,
                    task=request.task,
                    latency_ms=latency_ms,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )

                response = AgentResponse.create_error(
                    agent=self.name,
                    error_message=error_msg,
                    latency_ms=latency_ms
                )

                # Update trace with error info
                self.tracer.update_current_trace(
                    output=response.model_dump(),
                    tags=[self.name, request.task, "error"]
                )

                return response

    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True
    )
    async def _call_llm_with_timeout(
        self,
        model: Any,
        prompt: str,
        config: Optional[Any] = None
    ) -> Any:
        """
        Call LLM with timeout, rate limiting, and automatic retries.

        This method provides production-ready resilience:
        - Timeout: Prevents hanging on slow API responses
        - Rate limiting: Controls concurrent LLM calls via shared semaphore
        - Retries: Automatic exponential backoff on transient failures
        - Cost tracking: Logs tokens and estimated cost

        Args:
            model: GenerativeModel instance
            prompt: Prompt to send to LLM
            config: Optional GenerationConfig (for structured outputs)

        Returns:
            LLM response

        Raises:
            asyncio.TimeoutError: If call exceeds settings.llm_timeout (after retries)
            ConnectionError: If network fails (after retries)

        Example:
            response = await self._call_llm_with_timeout(
                self.model,
                "Classify this intent: ...",
                config
            )
        """
        # Rate limiting: use service-specific semaphore (max N concurrent calls)
        async with RateLimiters.llm:
            self.logger.info(
                "llm_call_started",
                agent=self.name,
                timeout=settings.llm_timeout,
                model=settings.agent_model,
                prompt_length=len(prompt),
                rate_limiter="llm_semaphore"
            )

            try:
                # Timeout protection: fail if LLM takes too long
                response = await asyncio.wait_for(
                    model.generate_content_async(prompt, generation_config=config),
                    timeout=settings.llm_timeout
                )

                # Track tokens usage for cost monitoring
                tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
                estimated_cost_usd = tokens_used * 0.00002  # Gemini Flash pricing

                self.logger.info(
                    "llm_call_completed",
                    agent=self.name,
                    tokens_used=tokens_used,
                    estimated_cost_usd=round(estimated_cost_usd, 6),
                    response_length=len(response.text) if hasattr(response, 'text') else 0
                )

                return response

            except asyncio.TimeoutError:
                self.logger.error(
                    "llm_call_timeout",
                    agent=self.name,
                    timeout=settings.llm_timeout,
                    timeout_seconds=settings.llm_timeout
                )
                raise

            except ConnectionError as e:
                self.logger.error(
                    "llm_call_connection_error",
                    agent=self.name,
                    error=str(e),
                    error_type="network_error"
                )
                raise

    @abstractmethod
    async def _execute_task(self, request: AgentRequest) -> Any:
        """
        Execute the specific task logic for this agent.

        This method MUST be implemented by all subclasses.

        Args:
            request: The validated agent request containing:
                - task: Task identifier (e.g., "search_policy", "get_order")
                - context: Task-specific parameters
                - metadata: Session and trace info

        Returns:
            Task result (can be dict, string, list, etc.)

        Raises:
            NotImplementedError: If subclass doesn't implement this
            Exception: Any task-specific errors (will be caught by handle_request)

        Example:
            async def _execute_task(self, request: AgentRequest) -> Any:
                if request.task == "search_policy":
                    query = request.context.get("query")
                    return await self._search_policy(query)
                else:
                    raise ValueError(f"Unknown task: {request.task}")
        """
        raise NotImplementedError(f"Agent {self.name} must implement _execute_task()")
