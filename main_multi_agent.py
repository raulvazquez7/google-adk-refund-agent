"""
Main entry point for the Multi-Agent Refund System (CLI) with auto-refund capability.

This script initializes and runs a command-line interface for interacting with
the multi-agent system, including:
- CoordinatorAgent (orchestrator)
- PolicyExpertAgent (RAG specialist)
- TransactionAgent (orders + refunds)

Features:
- Automatic eligibility checking
- User confirmation before processing refunds
- Persistent order context across conversation
"""
import asyncio
import uuid
import re
from dotenv import load_dotenv

# Load .env FIRST
load_dotenv()

from langfuse import Langfuse
from src.agents import (
    CoordinatorAgent,
    PolicyExpertAgent,
    TransactionAgent,
    AgentRequest
)
from src.models.schemas import AgentResponseTemplate
from src.utils.logger import get_logger
from src.utils.conversation_history import ConversationHistoryManager

# Initialize logger
logger = get_logger(__name__)

# Application constants
APP_NAME = "barefoot_multi_agent"
USER_ID = "local_test_user"
SESSION_ID = f"session_{uuid.uuid4()}"


def is_confirmation(text: str) -> bool:
    """
    Check if user text is a confirmation (yes, sí, confirmar, etc.).

    Args:
        text: User input

    Returns:
        True if confirmation detected
    """
    text_lower = text.lower().strip()
    confirmations = [
        "yes", "si", "sí", "ok", "confirmar", "confirmo",
        "proceder", "adelante", "vale", "afirmativo", "correcto"
    ]
    return any(conf in text_lower for conf in confirmations)


# Removed: extract_order_id() is now handled by CoordinatorAgent._extract_order_id()
# This eliminates code duplication and centralizes order ID extraction logic


async def main():
    """
    Main function to run a chat session with the multi-agent system.

    The flow is:
    User Input → Coordinator → [PolicyExpert, TransactionAgent] → Final Response
    """
    print("=" * 70)
    print("🤖 BAREFOOT ZÉNIT - MULTI-AGENT REFUND SYSTEM")
    print("=" * 70)
    print(f"Session ID: {SESSION_ID}")
    print("\n✨ Features:")
    print("  • Automatic refund eligibility checking")
    print("  • Confirms before processing refunds")
    print("  • Updates order status in database")
    print("\nType 'exit' to end the conversation.")
    print("Type 'help' for example queries.")
    print("-" * 70)

    # Initialize Langfuse tracer
    langfuse = Langfuse()

    # Step 1: Initialize specialized agents
    logger.info("system_initialization_started", session_id=SESSION_ID)

    print("\n[Initializing Agents...]")
    policy_expert = PolicyExpertAgent(tracer=langfuse)
    print(f"  ✅ {policy_expert.name}")

    transaction_agent = TransactionAgent(tracer=langfuse)
    print(f"  ✅ {transaction_agent.name}")

    # Step 2: Initialize coordinator
    coordinator = CoordinatorAgent(
        tracer=langfuse,
        specialized_agents={
            "policy_expert": policy_expert,
            "transaction_agent": transaction_agent
        }
    )
    print(f"  ✅ {coordinator.name}")

    logger.info(
        "system_initialization_completed",
        session_id=SESSION_ID,
        agents=["policy_expert", "transaction_agent", "coordinator"]
    )

    # Initialize conversation history manager
    history_manager = ConversationHistoryManager(
        max_tokens=16000,  # Gemini 2.0 Flash context window
        target_tokens=12000,  # Trigger compaction at 75%
        keep_recent_messages=8,  # Keep last 8 messages
        enable_summarization=True
    )

    print("\n✅ System ready!")
    print("-" * 70)

    # Session state
    pending_refund_order_id = None
    pending_refund_amount = None

    # Main conversation loop
    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                print("\n👋 Goodbye! Session ended.")
                break

            if user_input.lower() == 'help':
                print("\n📖 Example queries:")
                print("  • What is your refund policy?")
                print("  • Can I return shoes if I've tried them indoors?")
                print("  • I want to return my order ORD-84315")
                print("  • Do you accept refunds after 14 days?")
                continue

            # Add user message to history
            history_manager.add_message("user", user_input)

            # Check if user is confirming a pending refund
            if pending_refund_order_id and is_confirmation(user_input):
                print("\n✅ Processing refund...")

                # Call TransactionAgent to process refund
                refund_request = AgentRequest(
                    agent="transaction_agent",
                    task="process_refund",
                    context={
                        "order_id": pending_refund_order_id,
                        "amount": pending_refund_amount
                    },
                    metadata={"session_id": SESSION_ID}
                )

                refund_response = await transaction_agent.handle_request(refund_request)

                if refund_response.status == "success":
                    result = refund_response.result
                    if result.get("success"):
                        success_msg = f"✅ Refund processed successfully! Transaction ID: {result.get('transaction_id')}, Amount: ${result.get('amount')}"
                        print(f"\n{success_msg}")
                        print(f"   Order {pending_refund_order_id} status → RETURNED")
                        print(f"   Refund date: {result.get('refund_date')}")

                        # Add to history
                        history_manager.add_message("assistant", success_msg)
                    else:
                        error_msg = f"❌ Refund failed: {result.get('error')}"
                        print(f"\n{error_msg}")
                        history_manager.add_message("assistant", error_msg)
                else:
                    error_msg = f"❌ Error: {refund_response.error}"
                    print(f"\n{error_msg}")
                    history_manager.add_message("assistant", error_msg)

                # Clear pending refund
                pending_refund_order_id = None
                pending_refund_amount = None
                continue

            # Create request for coordinator
            request = AgentRequest(
                agent="coordinator",
                task="handle_user_query",
                context={"user_message": user_input},
                metadata={
                    "session_id": SESSION_ID,
                    "user_id": USER_ID
                }
            )

            # Execute with Langfuse tracing
            with langfuse.start_as_current_span(name="user-interaction"):
                langfuse.update_current_trace(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    input={"message": user_input},
                    tags=["multi-agent", "cli"]
                )

                print("\n🔄 Processing...\n")

                # Call coordinator
                response = await coordinator.handle_request(request)

                # Display response
                if response.status == "success":
                    result = response.result

                    # Show which agents were called
                    agents_called = result.get('agents_called', [])
                    if agents_called:
                        print(f"🔍 [Consulted: {', '.join(agents_called)}]")

                    # Extract structured response
                    response_template: AgentResponseTemplate = result.get('response')

                    # Display response type with emoji
                    response_type_emoji = {
                        "refund_eligible": "✅",
                        "refund_not_eligible": "❌",
                        "refund_already_processed": "ℹ️ ",
                        "policy_info": "📋",
                        "general_info": "💬",
                        "error": "⚠️ "
                    }
                    emoji = response_type_emoji.get(response_template.response_type, "🤖")
                    response_type_display = response_template.response_type.replace('_', ' ').title()

                    print(f"\n{emoji} {response_type_display}")
                    print("-" * 70)
                    print(f"\n{response_template.message}")

                    # Show key details if available
                    if response_template.key_details:
                        print("\n📌 Key Details:")
                        for detail in response_template.key_details:
                            print(f"  • {detail}")

                    # Show next action if available
                    if response_template.action_required:
                        print(f"\n➡️  Next Step: {response_template.action_required}")

                    print("-" * 70)

                    # Check if order is eligible for refund (uses eligibility_info from coordinator)
                    eligibility_info = result.get('eligibility_info')
                    final_response = response_template.message
                    extracted_order_id = result.get('extracted_order_id')  # From coordinator

                    # Only offer confirmation if eligible AND not already processed
                    if (eligibility_info and eligibility_info.eligible and
                        response_template.response_type == "refund_eligible" and
                        extracted_order_id):
                        # Order is eligible - offer confirmation
                        # No need to call TransactionAgent again, coordinator already did it
                        get_order_req = AgentRequest(
                            agent="transaction_agent",
                            task="get_order",
                            context={"order_id": extracted_order_id}
                        )
                        order_resp = await transaction_agent.handle_request(get_order_req)

                        if order_resp.status == "success" and order_resp.result.get("found"):
                            order_data = order_resp.result.get("order_data", {})
                            items = order_data.get("items", [])
                            total = sum(item.get("price", 0) for item in items)

                            pending_refund_order_id = extracted_order_id
                            pending_refund_amount = total

                            print(f"\n💡 Reply 'yes' to confirm refund of ${total:.2f} for order {extracted_order_id}")

                    # Add assistant response to history
                    history_manager.add_message(
                        "assistant",
                        response_template.message,
                        metadata={
                            "intent": result.get("intent"),
                            "response_type": response_template.response_type,
                            "agents_called": agents_called
                        }
                    )

                    # Show latency
                    latency = response.metadata.get('latency_ms', 0)
                    print(f"\n⏱️  Latency: {latency}ms")

                    # Show history stats
                    stats = history_manager.get_stats()
                    print(f"💾 Context: {stats['total_messages']} msgs, {stats['total_tokens']} tokens ({stats['token_usage_percent']:.1f}%)")

                else:
                    error_msg = f"❌ Error: {response.error}"
                    print(f"\n{error_msg}")
                    final_response = ""

                    # Add error to history
                    history_manager.add_message("assistant", error_msg)

                # Update trace
                langfuse.update_current_trace(
                    output={"response": final_response if response.status == "success" else response.error}
                )

                # Get trace URL
                trace_url = langfuse.get_trace_url()
                print(f"📊 Trace: {trace_url}")

            # Flush to Langfuse
            langfuse.flush()

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user.")
            break
        except Exception as e:
            logger.error("interaction_error", error=e, session_id=SESSION_ID)
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

    # Cleanup
    logger.info("session_ended", session_id=SESSION_ID)
    print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())
