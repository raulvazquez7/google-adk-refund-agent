"""
Coordinator Agent - The brain of the multi-agent system.

Routes user requests to specialized agents and orchestrates their responses.
"""
import asyncio
from typing import Dict, Any, List
from google.genai import types
from vertexai.generative_models import GenerativeModel

from src.agents.base_agent import BaseAgent
from src.protocols import AgentRequest, AgentResponse


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates specialized agents to handle user requests.
    
    Responsibilities:
    - Intent classification
    - Agent routing and delegation
    - Response assembly
    """
    
    def __init__(self, name: str, tracer, specialized_agents: Dict[str, BaseAgent], model_name: str = "gemini-2.5-flash"):
        """
        Initialize coordinator with specialized agents.
        
        Args:
            name: Agent identifier
            tracer: Langfuse client
            specialized_agents: Dict mapping agent names to instances
            model_name: LLM model for intent classification
        """
        super().__init__(name, tracer)
        self.agents = specialized_agents
        self.model = GenerativeModel(model_name)
        self.logger.info(f"Coordinator initialized with agents: {list(specialized_agents.keys())}")
    
    async def _execute_task(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Main orchestration logic.
        
        Flow:
        1. Classify user intent
        2. Route to appropriate agents
        3. Execute agent calls (parallel when possible)
        4. Assemble final response
        """
        user_message = request.context.get("user_message", "")
        
        # Step 1: Classify intent
        intent = await self._classify_intent(user_message)
        
        # Step 2: Determine which agents to call
        agent_calls = self._plan_agent_calls(intent, request.context)
        
        # Step 3: Execute calls
        results = await self._execute_agent_calls(agent_calls)
        
        # Step 4: Assemble response
        final_response = await self._assemble_response(intent, results, user_message)
        
        return {
            "intent": intent,
            "agents_called": list(results.keys()),
            "response": final_response
        }
    
    async def _classify_intent(self, user_message: str) -> str:
        """
        Classify user intent using LLM.
        
        Returns: refund, exchange, shipping, policy, general
        """
        prompt = f"""Classify the user's intent into ONE category:

Categories:
- refund: User wants to return product and get money back
- exchange: User wants different size/model
- shipping: User asks about delivery status
- policy: User asks about policies/rules
- general: Other questions

User message: "{user_message}"

Return ONLY the category name, nothing else."""

        response = await self.model.generate_content_async(prompt)
        intent = response.text.strip().lower()
        
        # Validate
        valid_intents = ["refund", "exchange", "shipping", "policy", "general"]
        return intent if intent in valid_intents else "general"
    
    def _plan_agent_calls(self, intent: str, context: Dict) -> List[Dict[str, Any]]:
        """
        Decide which agents to call based on intent.
        
        Returns list of agent call configs.
        """
        calls = []
        
        if intent == "refund":
            # Need policy + order details
            calls.append({
                "agent": "policy_expert",
                "task": "check_refund_policy",
                "context": {"query": "refund eligibility requirements"},
                "parallel": True
            })
            if context.get("order_id"):
                calls.append({
                    "agent": "transaction_agent",
                    "task": "get_order",
                    "context": {"order_id": context["order_id"]},
                    "parallel": True
                })
        
        elif intent == "exchange":
            # Need policy + order + stock
            calls.append({
                "agent": "policy_expert",
                "task": "check_exchange_policy",
                "context": {"query": "exchange policy"},
                "parallel": True
            })
            if context.get("order_id"):
                calls.append({
                    "agent": "transaction_agent",
                    "task": "get_order",
                    "context": {"order_id": context["order_id"]},
                    "parallel": True
                })
        
        elif intent == "shipping":
            if context.get("order_id"):
                calls.append({
                    "agent": "shipping_agent",
                    "task": "track_shipment",
                    "context": {"order_id": context["order_id"]},
                    "parallel": False
                })
        
        elif intent == "policy":
            calls.append({
                "agent": "policy_expert",
                "task": "search_policy",
                "context": {"query": context.get("user_message", "")},
                "parallel": False
            })
        
        return calls
    
    async def _execute_agent_calls(self, calls: List[Dict]) -> Dict[str, AgentResponse]:
        """
        Execute agent calls (parallel when possible).
        
        Returns dict mapping agent names to responses.
        """
        results = {}
        
        # Group by parallel flag
        parallel_calls = [c for c in calls if c.get("parallel", False)]
        sequential_calls = [c for c in calls if not c.get("parallel", False)]
        
        # Execute parallel calls
        if parallel_calls:
            tasks = []
            for call in parallel_calls:
                agent = self.agents.get(call["agent"])
                if agent:
                    request = AgentRequest(
                        agent=call["agent"],
                        task=call["task"],
                        context=call["context"]
                    )
                    tasks.append(agent.handle_request(request))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for call, response in zip(parallel_calls, responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Agent {call['agent']} failed: {response}")
                    results[call["agent"]] = AgentResponse.create_error(
                        agent=call["agent"],
                        error_message=str(response)
                    )
                else:
                    results[call["agent"]] = response
        
        # Execute sequential calls
        for call in sequential_calls:
            agent = self.agents.get(call["agent"])
            if agent:
                request = AgentRequest(
                    agent=call["agent"],
                    task=call["task"],
                    context=call["context"]
                )
                response = await agent.handle_request(request)
                results[call["agent"]] = response
        
        return results
    
    async def _assemble_response(self, intent: str, results: Dict[str, AgentResponse], user_message: str) -> str:
        """
        Assemble final response using LLM with agent results as context.
        """
        # Extract successful results
        context_parts = []
        for agent_name, response in results.items():
            if response.status == "success":
                context_parts.append(f"{agent_name}: {response.result}")
            else:
                context_parts.append(f"{agent_name}: ERROR - {response.error}")
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""You are a customer service agent for Barefoot Zénit.

User asked: "{user_message}"
Intent: {intent}

Information from specialized agents:
{context_str}

Provide a helpful, professional response based on this information.
If agents failed, apologize and explain you're having technical issues."""

        response = await self.model.generate_content_async(prompt)
        return response.text.strip()

