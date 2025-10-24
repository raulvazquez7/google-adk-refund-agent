"""
Quick test to verify conversation memory is working correctly.

This test simulates a multi-turn conversation to ensure:
1. History is being captured
2. History is being passed to the coordinator
3. The agent can reference previous context
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from src.agents import CoordinatorAgent, PolicyExpertAgent, TransactionAgent, AgentRequest
from src.utils.conversation_history import ConversationHistoryManager
from langfuse import Langfuse


async def test_memory():
    """Test multi-turn conversation with memory."""

    print("=" * 70)
    print("🧪 TESTING CONVERSATION MEMORY")
    print("=" * 70)

    # Initialize agents
    langfuse = Langfuse()
    policy_expert = PolicyExpertAgent(tracer=langfuse)
    transaction_agent = TransactionAgent(tracer=langfuse)
    coordinator = CoordinatorAgent(
        tracer=langfuse,
        specialized_agents={
            "policy_expert": policy_expert,
            "transaction_agent": transaction_agent
        }
    )

    # Initialize history manager
    history_manager = ConversationHistoryManager(
        max_tokens=16000,
        target_tokens=12000,
        keep_recent_messages=8,
        enable_summarization=True
    )

    print("\n✅ Agents initialized\n")

    # ========== TURN 1: User asks about refund with order_id ==========
    print("=" * 70)
    print("🧑 Turn 1: Quiero devolver mi pedido ORD-84315")
    print("=" * 70)

    user_input_1 = "Quiero devolver mi pedido ORD-84315"
    history_manager.add_message("user", user_input_1)

    request_1 = AgentRequest(
        agent="coordinator",
        task="handle_user_query",
        context={
            "user_message": user_input_1,
            "history": history_manager.get_context_for_llm()
        }
    )

    response_1 = await coordinator.handle_request(request_1)

    if response_1.status == "success":
        response_template = response_1.result.get('response')
        print(f"\n🤖 Assistant: {response_template.message}\n")
        history_manager.add_message("assistant", response_template.message)

        # Check if order_id was extracted
        extracted_order_id = response_1.result.get('extracted_order_id')
        print(f"✓ Order ID extracted: {extracted_order_id}")
    else:
        print(f"\n❌ Error: {response_1.error}")
        return

    # ========== TURN 2: Follow-up question without mentioning order_id ==========
    print("\n" + "=" * 70)
    print("🧑 Turn 2: Gracias, ¿y qué pasa si lo he usado?")
    print("=" * 70)
    print("(Testing if agent remembers ORD-84315 from previous context)")

    user_input_2 = "Gracias, ¿y qué pasa si lo he usado?"
    history_manager.add_message("user", user_input_2)

    # Get updated history (should now include Turn 1)
    conversation_context = history_manager.get_context_for_llm()

    print(f"\n📝 History being sent to LLM:")
    print("-" * 70)
    print(conversation_context[:500] + "..." if len(conversation_context) > 500 else conversation_context)
    print("-" * 70)

    request_2 = AgentRequest(
        agent="coordinator",
        task="handle_user_query",
        context={
            "user_message": user_input_2,
            "history": conversation_context
        }
    )

    response_2 = await coordinator.handle_request(request_2)

    if response_2.status == "success":
        response_template = response_2.result.get('response')
        print(f"\n🤖 Assistant: {response_template.message}\n")
        history_manager.add_message("assistant", response_template.message)

        # Check intent classification
        intent = response_2.result.get('intent')
        print(f"✓ Intent classified as: {intent}")

        # Success criteria: Should still understand this is about policy/refund context
        if intent in ["policy", "refund"]:
            print("\n✅ SUCCESS: Agent maintains conversation context!")
            print("   The follow-up question was correctly understood in context of the refund.")
        else:
            print("\n⚠️  WARNING: Intent classified as 'general'")
            print("   Agent may not have used conversation history effectively.")
    else:
        print(f"\n❌ Error: {response_2.error}")
        return

    # Show final stats
    print("\n" + "=" * 70)
    print("📊 CONVERSATION STATS")
    print("=" * 70)
    stats = history_manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_memory())
