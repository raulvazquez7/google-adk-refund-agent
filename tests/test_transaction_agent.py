"""
Quick test script for TransactionAgent.

This script demonstrates how to use TransactionAgent for order and refund operations.
"""
import asyncio
from dotenv import load_dotenv
from langfuse import Langfuse

from src.agents import TransactionAgent, AgentRequest

# Load environment variables
load_dotenv()


async def test_transaction_agent():
    """Test TransactionAgent with get_order and process_refund tasks."""
    print("=" * 60)
    print("Testing TransactionAgent")
    print("=" * 60)

    # Initialize Langfuse tracer
    langfuse = Langfuse()

    # Create TransactionAgent
    agent = TransactionAgent(tracer=langfuse)
    print(f"✅ Created agent: {agent.name}\n")

    # Test 1: Get Order Details
    print("-" * 60)
    print("TEST 1: Get Order Details")
    print("-" * 60)

    request = AgentRequest(
        agent="transaction_agent",
        task="get_order",
        context={"order_id": "ORD-84315"},
        metadata={"session_id": "test_session_123"}
    )

    print(f"📝 Request: task={request.task}, order_id=ORD-84315")
    response = await agent.handle_request(request)

    print(f"📊 Response:")
    print(f"  Status: {response.status}")
    print(f"  Latency: {response.metadata.get('latency_ms')}ms")

    if response.status == "success":
        result = response.result
        print(f"  ✅ Order found: {result.get('found')}")
        if result.get('found'):
            order_data = result.get('order_data', {})
            print(f"  Order ID: {order_data.get('order_id')}")
            print(f"  User ID: {order_data.get('user_id')}")
            print(f"  Status: {order_data.get('status')}")
            items = order_data.get('items', [])
            if items:
                print(f"  Items: {len(items)} product(s)")
                for item in items:
                    print(f"    - {item.get('name')}: ${item.get('price')}")
    else:
        print(f"  ❌ Error: {response.error}")

    # Test 2: Process Refund
    print("\n" + "-" * 60)
    print("TEST 2: Process Refund")
    print("-" * 60)

    request = AgentRequest(
        agent="transaction_agent",
        task="process_refund",
        context={
            "order_id": "ORD-84315",
            "amount": 89.99
        },
        metadata={"session_id": "test_session_123"}
    )

    print(f"📝 Request: task={request.task}, order_id=ORD-84315, amount=$89.99")
    response = await agent.handle_request(request)

    print(f"📊 Response:")
    print(f"  Status: {response.status}")
    print(f"  Latency: {response.metadata.get('latency_ms')}ms")

    if response.status == "success":
        result = response.result
        print(f"  ✅ Refund successful: {result.get('success')}")
        if result.get('success'):
            print(f"  Transaction ID: {result.get('transaction_id')}")
            print(f"  Amount: ${result.get('amount')}")
            print(f"  Message: {result.get('message')}")
    else:
        print(f"  ❌ Error: {response.error}")

    # Test 3: Error handling - Missing order_id
    print("\n" + "-" * 60)
    print("TEST 3: Error Handling (missing order_id)")
    print("-" * 60)

    request = AgentRequest(
        agent="transaction_agent",
        task="get_order",
        context={},  # Missing order_id
        metadata={"session_id": "test_session_123"}
    )

    print(f"📝 Request: task={request.task}, context={{}} (empty)")
    response = await agent.handle_request(request)

    print(f"📊 Response:")
    print(f"  Status: {response.status}")
    if response.status == "error":
        print(f"  ✅ Error handled correctly: {response.error}")

    # Flush Langfuse
    langfuse.flush()
    print("\n" + "=" * 60)
    print("🔍 View traces in Langfuse Cloud")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_transaction_agent())
