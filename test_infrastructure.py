"""
Quick test to validate BaseAgent infrastructure before building real agents.
"""
import asyncio
import os
from dotenv import load_dotenv
from langfuse import get_client

from src.protocols import AgentRequest, AgentResponse
from src.agents.test_agent import TestAgent

# Load environment
load_dotenv()


async def test_basic_flow():
    """Test 1: Basic request/response flow"""
    print("\n🧪 TEST 1: Basic Request/Response Flow")
    print("-" * 50)
    
    # Initialize
    tracer = get_client()
    agent = TestAgent(name="test_agent", tracer=tracer)
    
    # Create request
    request = AgentRequest(
        agent="test_agent",
        task="echo_test",
        context={"message": "Hello from test!"},
        metadata={"test_id": "001"}
    )
    
    # Execute
    response = await agent.handle_request(request)
    
    # Validate
    assert response.status == "success", "Response should be successful"
    assert response.agent == "test_agent", "Agent name should match"
    assert "message" in response.result, "Result should contain message"
    assert response.metadata["latency_ms"] > 0, "Should track latency"
    
    print(f"✅ Status: {response.status}")
    print(f"✅ Agent: {response.agent}")
    print(f"✅ Latency: {response.metadata['latency_ms']}ms")
    print(f"✅ Result: {response.result['message']}")
    
    tracer.flush()


async def test_error_handling():
    """Test 2: Error handling"""
    print("\n🧪 TEST 2: Error Handling")
    print("-" * 50)
    
    from src.agents.base_agent import BaseAgent
    
    # Create agent that always fails
    class FailingAgent(BaseAgent):
        async def _execute_task(self, request):
            raise ValueError("Intentional test error")
    
    tracer = get_client()
    agent = FailingAgent(name="failing_agent", tracer=tracer)
    
    request = AgentRequest(
        agent="failing_agent",
        task="fail_test",
        context={}
    )
    
    # Execute (should NOT crash, should return error response)
    response = await agent.handle_request(request)
    
    # Validate
    assert response.status == "error", "Should return error status"
    assert "Intentional test error" in response.error, "Error message should be preserved"
    assert response.result is None, "Result should be None on error"
    
    print(f"✅ Status: {response.status}")
    print(f"✅ Error captured: {response.error[:50]}...")
    print(f"✅ System didn't crash - error handled gracefully")
    
    tracer.flush()


async def test_tracing():
    """Test 3: Langfuse tracing"""
    print("\n🧪 TEST 3: Langfuse Tracing")
    print("-" * 50)
    
    tracer = get_client()
    agent = TestAgent(name="test_agent", tracer=tracer)
    
    # Execute with tracing
    with tracer.start_as_current_span(name="infrastructure_test"):
        tracer.update_current_trace(
            user_id="test_user",
            session_id="test_session",
            tags=["infrastructure", "validation"]
        )
        
        request = AgentRequest(
            agent="test_agent",
            task="tracing_test",
            context={"data": "test"}
        )
        
        response = await agent.handle_request(request)
        
        tracer.update_current_trace(output={"test_completed": True})
        
        # Get trace URL
        trace_url = tracer.get_trace_url()
        print(f"✅ Trace created successfully")
        print(f"🔍 View trace: {trace_url}")
    
    tracer.flush()
    print("✅ All traces sent to Langfuse")


async def test_multiple_agents():
    """Test 4: Multiple agent instances"""
    print("\n🧪 TEST 4: Multiple Agent Instances")
    print("-" * 50)
    
    tracer = get_client()
    
    # Create multiple agents
    agent1 = TestAgent(name="agent_1", tracer=tracer)
    agent2 = TestAgent(name="agent_2", tracer=tracer)
    
    # Execute in parallel
    request1 = AgentRequest(agent="agent_1", task="task_1", context={"id": 1})
    request2 = AgentRequest(agent="agent_2", task="task_2", context={"id": 2})
    
    responses = await asyncio.gather(
        agent1.handle_request(request1),
        agent2.handle_request(request2)
    )
    
    # Validate
    assert responses[0].agent == "agent_1"
    assert responses[1].agent == "agent_2"
    assert all(r.status == "success" for r in responses)
    
    print(f"✅ Agent 1 completed in {responses[0].metadata['latency_ms']}ms")
    print(f"✅ Agent 2 completed in {responses[1].metadata['latency_ms']}ms")
    print(f"✅ Multiple agents can run independently")
    
    tracer.flush()


async def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("🚀 Testing BaseAgent Infrastructure")
    print("=" * 50)
    
    try:
        await test_basic_flow()
        await test_error_handling()
        await test_tracing()
        await test_multiple_agents()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED - Infrastructure is solid!")
        print("=" * 50)
        print("\n✨ Ready to build real agents (Coordinator, PolicyExpert, etc.)")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
