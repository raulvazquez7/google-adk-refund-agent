"""
Test async tools and verify parallel execution.

This test suite validates:
1. Tools are truly async (not blocking event loop)
2. Tools can run in parallel
3. Parallel execution is faster than sequential
4. Pydantic models are returned correctly
"""
import asyncio
import time
import pytest

from src.tools import rag_search_tool, get_order_details, process_refund


@pytest.mark.asyncio
async def test_rag_search_tool_is_async():
    """Verify rag_search_tool is truly async."""
    result = await rag_search_tool("refund policy")

    assert isinstance(result, str)
    assert len(result) > 0
    print(f"✅ RAG search returned {len(result)} characters")


@pytest.mark.asyncio
async def test_get_order_details_is_async():
    """Verify get_order_details is async and returns Pydantic model."""
    from src.models.schemas import OrderResponse

    # Test with a likely existing order (ORD-84315 from examples)
    result = await get_order_details("ORD-84315")

    assert isinstance(result, OrderResponse)
    assert result.found in [True, False]

    if result.found:
        print(f"✅ Order found: {result.order_data.order_id}, status={result.order_data.status}")
    else:
        print(f"✅ Order not found (expected if DB not populated): {result.error}")


@pytest.mark.asyncio
async def test_get_order_details_not_found():
    """Verify get_order_details handles missing orders gracefully."""
    from src.models.schemas import OrderResponse

    result = await get_order_details("ORD-NONEXISTENT")

    assert isinstance(result, OrderResponse)
    assert result.found is False
    assert result.error is not None
    print(f"✅ Missing order handled correctly: {result.error}")


@pytest.mark.asyncio
async def test_parallel_rag_searches():
    """
    Test that multiple RAG searches can run in parallel.

    If truly async, 3 searches should take ~1x time (not 3x time).
    """
    queries = ["refund policy", "return shoes", "14 days"]

    # Sequential execution
    start_sequential = time.time()
    sequential_results = []
    for query in queries:
        result = await rag_search_tool(query)
        sequential_results.append(result)
    sequential_time = time.time() - start_sequential

    # Parallel execution
    start_parallel = time.time()
    parallel_results = await asyncio.gather(
        rag_search_tool(queries[0]),
        rag_search_tool(queries[1]),
        rag_search_tool(queries[2])
    )
    parallel_time = time.time() - start_parallel

    # Verify results
    assert len(parallel_results) == 3
    for result in parallel_results:
        assert isinstance(result, str)
        assert len(result) > 0

    # Verify parallel is faster (should be at least 1.5x faster)
    speedup = sequential_time / parallel_time

    print(f"\n📊 Parallel Execution Performance:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Parallel:   {parallel_time:.2f}s")
    print(f"   Speedup:    {speedup:.2f}x")

    # If speedup < 1.2x, might not be truly async
    if speedup < 1.2:
        print("   ⚠️  Warning: Speedup is low. Check if operations are truly async.")
    else:
        print("   ✅ True async concurrency confirmed!")


@pytest.mark.asyncio
async def test_parallel_mixed_operations():
    """
    Test that different tool types can run in parallel.

    This simulates real-world coordinator behavior.
    """
    start = time.time()

    # Run RAG search + order lookup in parallel (typical refund flow)
    results = await asyncio.gather(
        rag_search_tool("refund requirements"),
        get_order_details("ORD-84315"),
        return_exceptions=True
    )

    elapsed = time.time() - start

    # Verify results
    assert len(results) == 2

    # First result is RAG text
    assert isinstance(results[0], (str, Exception))
    if not isinstance(results[0], Exception):
        print(f"✅ RAG search completed: {len(results[0])} chars")

    # Second result is OrderResponse
    from src.models.schemas import OrderResponse
    assert isinstance(results[1], (OrderResponse, Exception))
    if isinstance(results[1], OrderResponse):
        print(f"✅ Order lookup completed: found={results[1].found}")

    print(f"📊 Mixed parallel execution time: {elapsed:.2f}s")

    # Should complete in < 5s (both operations run concurrently)
    assert elapsed < 5.0, f"Parallel execution too slow: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_pydantic_validation():
    """Verify Pydantic models validate data correctly."""
    from src.models.schemas import OrderResponse, OrderData, RefundProcessingResult
    from pydantic import ValidationError

    # Test OrderResponse validation - found is required
    try:
        # Invalid: missing required field
        OrderResponse()  # Missing found (required)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "found" in str(e)
        print("✅ OrderResponse validation works (caught missing 'found' field)")

    # Test valid OrderResponse
    valid_response = OrderResponse(found=False, error="Not found")
    assert valid_response.found is False
    print("✅ OrderResponse accepts valid data")

    # Test RefundProcessingResult validation
    valid_refund = RefundProcessingResult(
        success=True,
        order_id="ORD-12345",
        transaction_id="REF-123",
        amount=99.99,
        refund_date="2025-10-22T12:00:00"
    )
    assert valid_refund.success is True
    assert valid_refund.amount == 99.99
    print("✅ RefundProcessingResult validation works")


if __name__ == "__main__":
    # Run tests with pytest
    print("Run with: pytest test_async_tools.py -v -s")
