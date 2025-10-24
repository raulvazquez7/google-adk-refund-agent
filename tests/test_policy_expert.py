"""
Quick test script for PolicyExpertAgent.

This script demonstrates how to use PolicyExpertAgent standalone.
"""
import asyncio
from dotenv import load_dotenv
from langfuse import Langfuse

from src.agents import PolicyExpertAgent, AgentRequest

# Load environment variables
load_dotenv()


async def test_policy_expert():
    """Test PolicyExpertAgent with a sample query."""
    print("=" * 60)
    print("Testing PolicyExpertAgent")
    print("=" * 60)

    # Initialize Langfuse tracer
    langfuse = Langfuse()

    # Create PolicyExpertAgent
    agent = PolicyExpertAgent(tracer=langfuse)
    print(f"✅ Created agent: {agent.name}")

    # Create request
    request = AgentRequest(
        agent="policy_expert",
        task="search_policy",
        context={
            "query": "Can I return shoes if I've tried them indoors?"
        },
        metadata={
            "session_id": "test_session_123",
            "user_id": "test_user"
        }
    )

    print(f"\n📝 Request:")
    print(f"  Task: {request.task}")
    print(f"  Query: {request.context.get('query')}")

    # Execute request
    print(f"\n🔍 Executing RAG search...")
    response = await agent.handle_request(request)

    # Display results
    print(f"\n📊 Response:")
    print(f"  Status: {response.status}")
    print(f"  Agent: {response.agent}")
    print(f"  Latency: {response.metadata.get('latency_ms')}ms")

    if response.status == "success":
        result = response.result
        print(f"\n✅ Success!")
        print(f"  Source: {result.get('source')}")
        print(f"  Policy text preview:")
        policy_text = result.get('policy_text', '')
        print(f"    {policy_text[:200]}...")
        print(f"    (Total length: {len(policy_text)} chars)")
    else:
        print(f"\n❌ Error: {response.error}")

    # Flush Langfuse
    langfuse.flush()
    print(f"\n🔍 View trace in Langfuse Cloud")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_policy_expert())
