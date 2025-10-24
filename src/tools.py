"""
Tool implementations for the Barefoot Zénit refund agent.

This module provides the concrete implementations of the tools that the agent
can use to perform its tasks. These tools interact with Firestore and Vertex AI
to gather information and execute actions.

Available tools:
- rag_search_tool: Performs semantic search on the refund policy.
- get_order_details: Retrieves specific order information from Firestore.
- process_refund: Simulates processing a refund and returns a transaction ID.
"""
import asyncio
import hashlib
from collections import OrderedDict
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

from google.cloud.firestore import AsyncClient
from vertexai.language_models import TextEmbeddingModel

from src.config import settings
from src.utils.logger import get_logger
from src.utils.rate_limiters import RateLimiters


logger = get_logger(__name__)


class EmbeddingsCache:
    """
    Thread-safe LRU cache for embeddings with hit/miss metrics.

    This cache significantly reduces API calls and costs for repeated queries.
    Cache key is based on normalized text hash.

    Metrics:
    - Cache hits: Queries served from cache
    - Cache misses: Queries requiring API calls
    - Cost savings: Estimated USD saved

    Example:
        cache = EmbeddingsCache(max_size=100)
        embedding = await cache.get_or_compute("user query", compute_fn)
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize embeddings cache.

        Args:
            max_size: Maximum number of cached embeddings (LRU eviction)
        """
        self._cache: OrderedDict[str, NDArray[np.float64]] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._cost_per_embedding = 0.0001  # ~$0.0001 per embedding call

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for cache key consistency.

        Args:
            text: Raw input text

        Returns:
            Normalized text (lowercase, stripped)
        """
        return text.lower().strip()

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key from text.

        Uses SHA-256 hash of normalized text for consistent keys.

        Args:
            text: Input text

        Returns:
            Cache key (hex digest)
        """
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    async def get(self, text: str) -> Optional[NDArray[np.float64]]:
        """
        Get embedding from cache if exists.

        Args:
            text: Query text

        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._get_cache_key(text)

        async with self._lock:
            if cache_key in self._cache:
                # Move to end (LRU: most recently used)
                self._cache.move_to_end(cache_key)
                self._hits += 1

                logger.info(
                    "embeddings_cache_hit",
                    cache_key=cache_key[:16],
                    total_hits=self._hits,
                    total_misses=self._misses,
                    hit_rate=f"{self.hit_rate:.2%}"
                )

                return self._cache[cache_key]

            self._misses += 1
            return None

    async def set(self, text: str, embedding: NDArray[np.float64]) -> None:
        """
        Store embedding in cache with LRU eviction.

        Args:
            text: Query text
            embedding: Embedding vector to cache
        """
        cache_key = self._get_cache_key(text)

        async with self._lock:
            # Add to cache
            self._cache[cache_key] = embedding
            self._cache.move_to_end(cache_key)

            # LRU eviction: remove oldest if over max_size
            if len(self._cache) > self._max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]

                logger.debug(
                    "embeddings_cache_eviction",
                    evicted_key=evicted_key[:16],
                    cache_size=len(self._cache)
                )

    async def get_or_compute(
        self,
        text: str,
        compute_fn: Any  # Callable that returns embeddings
    ) -> NDArray[np.float64]:
        """
        Get from cache or compute if missing (cache-aside pattern).

        Args:
            text: Query text
            compute_fn: Async function to compute embeddings if cache miss

        Returns:
            Embedding vector
        """
        # Try cache first
        cached = await self.get(text)
        if cached is not None:
            return cached

        # Cache miss - compute
        embedding = await compute_fn([text])
        embedding_vector = embedding[0]  # get_embeddings_async returns list

        # Store in cache
        await self.set(text, embedding_vector)

        logger.info(
            "embeddings_cache_miss",
            total_misses=self._misses,
            hit_rate=f"{self.hit_rate:.2%}",
            estimated_savings_usd=f"${self.estimated_savings:.4f}"
        )

        return embedding_vector

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def estimated_savings(self) -> float:
        """Estimate cost savings from cache (USD)."""
        return self._hits * self._cost_per_embedding

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.

        Returns:
            Dict with hits, misses, hit_rate, savings
        """
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "estimated_savings_usd": f"${self.estimated_savings:.4f}",
            "cache_size": len(self._cache),
            "max_size": self._max_size
        }


# Global embeddings cache instance
_embeddings_cache = EmbeddingsCache(max_size=settings.embeddings_cache_size)

# Initialize AsyncClient for true async Firestore operations
db = AsyncClient(
    project=settings.gcp_project_id,
    database=settings.firestore_database_id
)


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def _get_embeddings_async(texts: List[str]) -> List[NDArray[np.float64]]:
    """
    Generate embeddings asynchronously using VertexAI with rate limiting.

    Uses get_embeddings_async() for true async I/O without blocking event loop.
    Rate limited to prevent API saturation (10 concurrent calls by default).

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors

    Raises:
        Exception: If embedding generation fails
    """
    async with RateLimiters.embeddings:
        model = TextEmbeddingModel.from_pretrained(settings.embeddings_model)
        embeddings = await model.get_embeddings_async(texts)
        return [np.array(emb.values) for emb in embeddings]


async def _retrieve_policy_chunks_async() -> List[Dict[str, Any]]:
    """
    Retrieve all policy chunks from Firestore asynchronously.

    Uses AsyncClient.stream() which returns an async generator.

    Returns:
        List of policy chunks with embeddings

    Raises:
        Exception: If Firestore query fails
    """
    collection_ref = db.collection("policy_chunks")
    docs = collection_ref.stream()

    chunks = []
    async for doc in docs:
        data = doc.to_dict()
        chunks.append({
            "text": data["text"],
            "embedding": np.array(data["embedding"]),
            "chunk_id": data["chunk_id"]
        })

    return chunks


def _rank_chunks_by_similarity(
    query_vector: NDArray[np.float64],
    chunks: List[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Rank chunks by cosine similarity to query vector.

    Args:
        query_vector: Query embedding vector
        chunks: List of policy chunks with embeddings
        top_k: Number of top results to return

    Returns:
        Top K chunks sorted by similarity score
    """
    results = []
    for chunk in chunks:
        similarity = cosine_similarity(query_vector, chunk["embedding"])
        results.append({
            "text": chunk["text"],
            "similarity": similarity,
            "chunk_id": chunk["chunk_id"]
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


async def rag_search_tool(query: str) -> str:
    """
    Perform async semantic search on the company's refund policy.

    Now with embeddings caching for cost optimization!
    - Repeated queries are served from cache (no API call)
    - Cache metrics logged for observability

    Uses async I/O for embeddings generation and Firestore queries.
    This is production-ready for small-to-medium datasets.
    For millions of vectors, consider Vertex AI Vector Search.

    Args:
        query: User's search query

    Returns:
        Relevant policy sections (top K chunks concatenated)

    Raises:
        RuntimeError: If search fails
    """
    logger.info("rag_search_started", query=query)

    try:
        # Generate query embedding with caching (major cost optimization!)
        query_vector = await _embeddings_cache.get_or_compute(
            query,
            _get_embeddings_async
        )

        # Retrieve all policy chunks (async)
        chunks = await _retrieve_policy_chunks_async()

        if not chunks:
            logger.warning("rag_search_no_chunks", query=query)
            return "No policy information available in the database."

        # Rank chunks by similarity (sync operation, CPU-bound)
        top_results = _rank_chunks_by_similarity(
            query_vector,
            chunks,
            top_k=settings.rag_top_k
        )

        if not top_results:
            logger.warning("rag_search_no_results", query=query)
            return "No relevant information found in the refund policy."

        # Log success with cache metrics
        similarities = [f"{r['similarity']:.3f}" for r in top_results]
        cache_metrics = _embeddings_cache.get_metrics()

        logger.info(
            "rag_search_completed",
            query=query,
            num_results=len(top_results),
            similarities=similarities,
            cache_metrics=cache_metrics
        )

        # Return concatenated text
        context_pieces = [r["text"] for r in top_results]
        return "\n---\n".join(context_pieces)

    except Exception as e:
        logger.error("rag_search_failed", error=e, query=query)
        raise RuntimeError(f"RAG search failed for query '{query}': {str(e)}") from e


async def get_order_details(order_id: str) -> "OrderResponse":
    """
    Retrieve order details from Firestore asynchronously.

    Returns Pydantic model instead of JSON string for type safety.

    Args:
        order_id: The unique identifier for the order (e.g., "ORD-12345")

    Returns:
        OrderResponse with found=True and order_data if exists, else found=False with error

    Raises:
        Never raises - errors are captured in OrderResponse.error
    """
    from src.models.schemas import OrderResponse, OrderData

    logger.info("get_order_started", order_id=order_id)

    try:
        doc_ref = db.collection("orders").document(order_id)
        doc = await doc_ref.get()

        if not doc.exists:
            logger.warning("order_not_found", order_id=order_id)
            return OrderResponse(
                found=False,
                error=f"Order '{order_id}' not found in database."
            )

        order_data_dict = doc.to_dict()

        # Pydantic will handle datetime conversion automatically
        order_data = OrderData(**order_data_dict)

        logger.info("get_order_completed", order_id=order_id, status=order_data.status)
        return OrderResponse(found=True, order_data=order_data)

    except Exception as e:
        logger.error("get_order_failed", error=e, order_id=order_id)
        return OrderResponse(
            found=False,
            error=f"Failed to fetch order: {str(e)}"
        )


async def process_refund(order_id: str, amount: float) -> "RefundProcessingResult":
    """
    Process refund asynchronously and update Firestore.

    Returns Pydantic model instead of JSON string for type safety.

    This function:
    1. Validates order exists
    2. Checks if already refunded
    3. Updates order status to "RETURNED" in Firestore (async)
    4. Records refund timestamp
    5. Returns structured result

    Args:
        order_id: The order ID to be refunded
        amount: The amount to refund

    Returns:
        RefundProcessingResult with success status and details

    Raises:
        Never raises - errors are captured in RefundProcessingResult.error
    """
    from src.models.schemas import RefundProcessingResult

    logger.info("process_refund_started", order_id=order_id, amount=amount)

    try:
        # Get order from Firestore (async)
        doc_ref = db.collection("orders").document(order_id)
        doc = await doc_ref.get()

        if not doc.exists:
            logger.warning("process_refund_order_not_found", order_id=order_id)
            return RefundProcessingResult(
                success=False,
                order_id=order_id,
                error=f"Order {order_id} not found in database."
            )

        order_data = doc.to_dict()
        current_status = order_data.get("status")

        # Check if already returned
        if current_status == "RETURNED":
            refund_date = order_data.get("refund_date")
            refund_transaction_id = order_data.get("refund_transaction_id")
            refund_amount = order_data.get("refund_amount")

            logger.warning(
                "process_refund_already_returned",
                order_id=order_id,
                refund_date=refund_date,
                refund_transaction_id=refund_transaction_id
            )

            return RefundProcessingResult(
                success=False,
                order_id=order_id,
                error=f"Order {order_id} was already refunded on {refund_date}.",
                transaction_id=refund_transaction_id,
                refund_date=refund_date,
                amount=refund_amount
            )

        # Generate transaction ID and timestamp
        transaction_id = f"REF-{int(datetime.now().timestamp() * 1000)}"
        refund_timestamp = datetime.now().isoformat()

        # Update order in Firestore (async)
        await doc_ref.update({
            "status": "RETURNED",
            "refund_transaction_id": transaction_id,
            "refund_date": refund_timestamp,
            "refund_amount": amount
        })

        logger.info(
            "process_refund_completed",
            order_id=order_id,
            amount=amount,
            transaction_id=transaction_id,
            refund_date=refund_timestamp
        )

        return RefundProcessingResult(
            success=True,
            order_id=order_id,
            transaction_id=transaction_id,
            amount=amount,
            refund_date=refund_timestamp
        )

    except Exception as e:
        logger.error("process_refund_error", error=e, order_id=order_id)
        return RefundProcessingResult(
            success=False,
            order_id=order_id,
            error=f"Failed to process refund: {str(e)}"
        )
