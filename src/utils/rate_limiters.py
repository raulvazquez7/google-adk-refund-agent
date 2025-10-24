"""
Centralized rate limiting for all external services.

This module provides service-specific semaphores to control concurrent API calls
and prevent saturation of different backends.

Rate limiting strategy:
- LLM calls: Conservative (5 concurrent) - expensive, slow
- Embeddings: Moderate (10 concurrent) - faster than LLM
- Firestore: Generous (20 concurrent) - fast, can handle high load
"""
import asyncio
from typing import Optional

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiters:
    """
    Centralized rate limiters for external services.

    Each service gets its own semaphore to control concurrency independently.
    This prevents slow services (LLM) from blocking fast services (Firestore).

    Example:
        async with RateLimiters.llm:
            response = await model.generate_content_async(...)
    """

    # Class-level semaphores (shared across all instances)
    _llm_semaphore: Optional[asyncio.Semaphore] = None
    _embeddings_semaphore: Optional[asyncio.Semaphore] = None
    _firestore_semaphore: Optional[asyncio.Semaphore] = None
    _initialized = False

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize all rate limiters (called once at startup).

        This creates semaphores based on settings configuration.
        """
        if cls._initialized:
            return

        cls._llm_semaphore = asyncio.Semaphore(settings.llm_rate_limit)
        cls._embeddings_semaphore = asyncio.Semaphore(settings.embeddings_rate_limit)
        cls._firestore_semaphore = asyncio.Semaphore(settings.firestore_rate_limit)

        logger.info(
            "rate_limiters_initialized",
            llm_limit=settings.llm_rate_limit,
            embeddings_limit=settings.embeddings_rate_limit,
            firestore_limit=settings.firestore_rate_limit
        )

        cls._initialized = True


# Initialize on module import
RateLimiters.initialize()

# Expose as module-level variables for clean syntax: RateLimiters.llm
RateLimiters.llm = RateLimiters._llm_semaphore
RateLimiters.embeddings = RateLimiters._embeddings_semaphore
RateLimiters.firestore = RateLimiters._firestore_semaphore
