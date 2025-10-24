"""
End-to-end test of the multi-agent system.

This script demonstrates the complete flow:
User Query → Coordinator → Specialized Agents → Final Response
"""
import asyncio
from dotenv import load_dotenv
from langfuse import Langfuse

from src.agents import (
    CoordinatorAgent,
    PolicyExpertAgent,
    TransactionAgent,
    AgentRequest
)

# Load environment variables
load_dotenv()


async def test_multi_agent_system():
    """Test the complete multi-agent system end-to-end."""
    print("=" * 70)
    print("MULTI-AGENT SYSTEM END-TO-END TEST")
    print("=" * 70)

    # Initialize Langfuse tracer
    langfuse = Langfuse()

    # Step 1: Initialize specialized agents
    print("\n[STEP 1] Initializing specialized agents...")
    policy_expert = PolicyExpertAgent(tracer=langfuse)
    transaction_agent = TransactionAgent(tracer=langfuse)

    print(f"  ✅ {policy_expert.name}")
    print(f"  ✅ {transaction_agent.name}")

    # Step 2: Initialize coordinator
    print("\n[STEP 2] Initializing coordinator...")
    coordinator = CoordinatorAgent(
        tracer=langfuse,
        specialized_agents={
            "policy_expert": policy_expert,
            "transaction_agent": transaction_agent
        }
    )
    print(f"  ✅ {coordinator.name}")

    # Test scenarios
    scenarios = [
        {
            "name": "SCENARIO 1: Policy Question",
            "user_message": "Can I return shoes if I've only tried them indoors?",
            "expected_intent": "policy"
        },
        {
            "name": "SCENARIO 2: Refund Request with Order ID",
            "user_message": "I want to return my order ORD-84315. Can I get a refund?",
            "expected_intent": "refund"
        },
        {
            "name": "SCENARIO 3: General Refund Question",
            "user_message": "What is your refund policy?",
            "expected_intent": "policy"
        }
    ]

    for idx, scenario in enumerate(scenarios, 1):
        print("\n" + "=" * 70)
        print(f"[{scenario['name']}]")
        print("=" * 70)

        user_message = scenario["user_message"]
        print(f"\n📝 User Query:")
        print(f"  \"{user_message}\"")

        # Create request
        request = AgentRequest(
            agent="coordinator",
            task="handle_user_query",
            context={"user_message": user_message},
            metadata={"test_scenario": idx}
        )

        print(f"\n🔄 Processing...")

        # Execute
        response = await coordinator.handle_request(request)

        # Display results
        print(f"\n📊 Coordinator Response:")
        print(f"  Status: {response.status}")
        print(f"  Latency: {response.metadata.get('latency_ms')}ms")

        if response.status == "success":
            result = response.result
            print(f"\n  Detected Intent: {result.get('intent')}")
            print(f"  Agents Called: {', '.join(result.get('agents_called', []))}")
            print(f"\n  🤖 Final Response:")
            print(f"  {'-' * 66}")
            final_response = result.get('response', '')
            # Wrap text at 66 chars
            lines = final_response.split('\n')
            for line in lines:
                if len(line) <= 66:
                    print(f"  {line}")
                else:
                    words = line.split()
                    current_line = "  "
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 68:
                            current_line += word + " "
                        else:
                            print(current_line.rstrip())
                            current_line = "  " + word + " "
                    if current_line.strip():
                        print(current_line.rstrip())
            print(f"  {'-' * 66}")

        else:
            print(f"\n  ❌ Error: {response.error}")

        # Pause between scenarios
        if idx < len(scenarios):
            await asyncio.sleep(1)

    # Flush Langfuse
    print("\n" + "=" * 70)
    print("Flushing traces to Langfuse Cloud...")
    langfuse.flush()

    print("\n✅ ALL TESTS COMPLETED")
    print("🔍 View detailed traces in Langfuse Cloud")
    print("=" * 70)


async def test_parallel_execution():
    """
    Demonstrate parallel agent execution.

    When a refund request includes an order ID, the coordinator calls:
    - PolicyExpertAgent (search refund policy)
    - TransactionAgent (get order details)

    These execute IN PARALLEL using asyncio.gather for speed.
    """
    print("\n" + "=" * 70)
    print("BONUS: PARALLEL EXECUTION DEMO")
    print("=" * 70)

    langfuse = Langfuse()

    # Initialize agents
    policy_expert = PolicyExpertAgent(tracer=langfuse)
    transaction_agent = TransactionAgent(tracer=langfuse)

    coordinator = CoordinatorAgent(
        tracer=langfuse,
        specialized_agents={
            "policy_expert": policy_expert,
            "transaction_agent": transaction_agent
        }
    )

    print("\n📝 User Query (with Order ID):")
    print("  \"I need a refund for order ORD-84315\"")

    request = AgentRequest(
        agent="coordinator",
        task="handle_user_query",
        context={"user_message": "I need a refund for order ORD-84315"},
        metadata={"test": "parallel_execution"}
    )

    import time
    start = time.time()

    print("\n🚀 Executing with PARALLEL agent calls...")
    response = await coordinator.handle_request(request)

    elapsed = int((time.time() - start) * 1000)

    if response.status == "success":
        result = response.result
        print(f"\n✅ Completed in {elapsed}ms")
        print(f"  Agents called in parallel: {', '.join(result.get('agents_called', []))}")
        print(f"  Intent: {result.get('intent')}")
        print(f"\n  💡 Note: Without parallelization, this would take ~2x longer")
        print(f"     (each agent would run sequentially)")

    langfuse.flush()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_multi_agent_system())
    asyncio.run(test_parallel_execution())
