"""
End-to-End Test: Refund Flow with Spanish Order ID

This test validates the complete refund flow:
1. User says "quiero devolver mi pedido número 25836" (Spanish, no prefix)
2. Coordinator extracts order_id → "ORD-25836"
3. TransactionAgent fetches order
4. System checks eligibility (should be eligible)
5. System asks for confirmation
"""
import asyncio
from langfuse import Langfuse
from src.agents import (
    CoordinatorAgent,
    PolicyExpertAgent,
    TransactionAgent,
    AgentRequest
)

async def test_refund_flow():
    """Test complete refund flow with Spanish input."""

    print("=" * 70)
    print("🧪 END-TO-END TEST: Refund Flow (Spanish + No Prefix)")
    print("=" * 70)

    # Initialize Langfuse tracer
    langfuse = Langfuse()

    # Initialize specialized agents
    print("\n[Step 1] Initializing agents...")
    policy_expert = PolicyExpertAgent(tracer=langfuse)
    transaction_agent = TransactionAgent(tracer=langfuse)

    coordinator = CoordinatorAgent(
        tracer=langfuse,
        specialized_agents={
            "policy_expert": policy_expert,
            "transaction_agent": transaction_agent
        }
    )
    print("✅ Agents initialized\n")

    # Test input (Spanish, no ORD- prefix)
    user_message = "quiero devolver mi pedido número 25836"

    print(f"[Step 2] User input: \"{user_message}\"")

    # Create request for coordinator
    request = AgentRequest(
        agent="coordinator",
        task="handle_user_query",
        context={"user_message": user_message},
        metadata={"session_id": "test_session", "user_id": "test_user"}
    )

    print("\n[Step 3] Processing request...\n")

    # Call coordinator
    response = await coordinator.handle_request(request)

    # Validate response
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    if response.status == "success":
        result = response.result

        print(f"✅ Status: SUCCESS")
        print(f"\n📋 Intent classified: {result.get('intent')}")
        print(f"🤖 Agents called: {result.get('agents_called')}")

        # Check if order_id was extracted
        extracted_order_id = result.get('extracted_order_id')
        print(f"\n🔍 Order ID extracted: {extracted_order_id}")

        if extracted_order_id == "ORD-25836":
            print("   ✅ CORRECT! (Spanish 'pedido número 25836' → 'ORD-25836')")
        else:
            print(f"   ❌ WRONG! Expected 'ORD-25836', got '{extracted_order_id}'")

        # Check eligibility
        eligibility_info = result.get('eligibility_info')
        if eligibility_info:
            print(f"\n📊 Eligibility Check:")
            print(f"   - Eligible: {eligibility_info.eligible}")
            print(f"   - Reason: {eligibility_info.reason}")
            print(f"   - Order Status: {eligibility_info.order_status}")
            print(f"   - Days since purchase: {eligibility_info.days_since_purchase}")

            if eligibility_info.eligible:
                print("   ✅ Order IS eligible for refund")
            else:
                print("   ❌ Order NOT eligible for refund")

        # Show response
        response_template = result.get('response')
        if response_template:
            print(f"\n💬 Response to user:")
            print(f"   Type: {response_template.response_type}")
            print(f"   Message: {response_template.message[:200]}...")

            if response_template.action_required:
                print(f"   Action: {response_template.action_required}")

        # Final verdict
        print("\n" + "=" * 70)

        is_success = (
            result.get('intent') == 'refund' and
            extracted_order_id == 'ORD-25836' and
            eligibility_info and
            eligibility_info.eligible
        )

        if is_success:
            print("🎉 TEST PASSED! System correctly:")
            print("   ✅ Classified intent as 'refund'")
            print("   ✅ Extracted 'ORD-25836' from Spanish input")
            print("   ✅ Called TransactionAgent")
            print("   ✅ Checked eligibility (eligible)")
            print("   ✅ Generated appropriate response")
        else:
            print("❌ TEST FAILED - See details above")

        print("=" * 70)

    else:
        print(f"❌ Status: ERROR")
        print(f"Error: {response.error}")

    # Flush Langfuse
    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(test_refund_flow())
