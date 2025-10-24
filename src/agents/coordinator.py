"""
Coordinator Agent - The brain of the multi-agent system.

Routes user requests to specialized agents and orchestrates their responses.
"""
import asyncio
from typing import Dict, Any, List
from pydantic import ValidationError
from vertexai.generative_models import GenerativeModel, GenerationConfig

from src.agents.base_agent import BaseAgent
from src.models.protocols import AgentRequest, AgentResponse
from src.models.schemas import IntentClassification, AgentResponseTemplate, RefundEligibilityInfo
from src.config import settings
from src.utils.prompts import get_prompt


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates specialized agents to handle user requests.

    Responsibilities:
    - Intent classification using LLM
    - Agent routing and delegation
    - Parallel execution when possible
    - Response assembly

    Example:
        # Initialize with specialized agents
        agents = {
            "policy_expert": PolicyExpertAgent(tracer=langfuse),
            "transaction_agent": TransactionAgent(tracer=langfuse)
        }
        coordinator = CoordinatorAgent(tracer=langfuse, specialized_agents=agents)

        # Handle user request
        request = AgentRequest(
            agent="coordinator",
            task="handle_user_query",
            context={"user_message": "Can I return my order ORD-84315?"}
        )
        response = await coordinator.handle_request(request)
    """

    def __init__(self, tracer, specialized_agents: Dict[str, BaseAgent]):
        """
        Initialize coordinator with specialized agents.

        Args:
            tracer: Langfuse client for observability
            specialized_agents: Dict mapping agent names to instances
                Example: {"policy_expert": PolicyExpertAgent(...), ...}
        """
        super().__init__(name="coordinator", tracer=tracer)
        self.agents = specialized_agents
        self.model = GenerativeModel(settings.agent_model)

        agent_names = list(specialized_agents.keys())
        self.logger.info(
            "coordinator_initialized",
            agent=self.name,
            specialized_agents=agent_names,
            num_agents=len(agent_names)
        )

    async def _execute_task(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Main orchestration logic.

        Flow:
        1. Classify user intent
        2. Route to appropriate agents
        3. Execute agent calls (parallel when possible)
        4. Assemble final response

        Args:
            request: Request with context containing "user_message"

        Returns:
            Dict with intent, agents_called, and final response
        """
        user_message = request.context.get("user_message", "")

        if not user_message:
            raise ValueError("Missing 'user_message' in context")

        self.logger.info(
            "coordination_started",
            agent=self.name,
            user_message=user_message[:100]  # Log first 100 chars
        )

        # Step 1: Classify intent
        intent = await self._classify_intent(user_message)

        # Step 2: Determine which agents to call
        agent_calls = self._plan_agent_calls(intent, request.context)

        # Step 3: Execute calls
        results = await self._execute_agent_calls(agent_calls)

        # Step 4: Assemble response (returns dict with response + eligibility_info)
        response_data = await self._assemble_response(intent, results, user_message)

        self.logger.info(
            "coordination_completed",
            agent=self.name,
            intent=intent,
            agents_called=list(results.keys()),
            num_agents_called=len(results)
        )

        result = {
            "intent": intent,
            "agents_called": list(results.keys()),
            "response": response_data["response"]
        }

        # Include eligibility_info if available (for refund flow)
        if response_data.get("eligibility_info"):
            result["eligibility_info"] = response_data["eligibility_info"]

        # Include extracted order_id for refund confirmation (if intent was refund)
        if intent == "refund" and "transaction_agent" in results:
            trans_result = results["transaction_agent"].result
            if isinstance(trans_result, dict):
                extracted_order_id = trans_result.get("order_id")
                if extracted_order_id:
                    result["extracted_order_id"] = extracted_order_id

        return result

    async def _classify_intent(self, user_message: str) -> str:
        """
        Classify user intent using LLM with structured output.

        Always uses LLM for intent classification to ensure accurate
        understanding of user intent. This approach prioritizes correctness
        over optimization, preventing potential misclassification from
        rule-based shortcuts.

        Uses Pydantic IntentClassification schema for validated LLM outputs.

        Args:
            user_message: User's query

        Returns:
            Intent category: "refund", "policy", or "general"
        """
        self.logger.info(
            "intent_classification_started",
            agent=self.name,
            method="llm"
        )

        prompt = get_prompt("intent_classification", user_message=user_message)

        config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=IntentClassification.model_json_schema()
        )

        try:
            response = await self._call_llm_with_timeout(self.model, prompt, config)
            classification = IntentClassification.model_validate_json(response.text)

            self.logger.info(
                "intent_classification_completed",
                agent=self.name,
                intent=classification.intent,
                confidence=classification.confidence
            )

            return classification.intent

        except ValidationError as e:
            self.logger.error(
                "intent_classification_validation_failed",
                agent=self.name,
                error=str(e),
                defaulting_to="general"
            )
            return "general"

    def _plan_agent_calls(self, intent: str, context: Dict) -> List[Dict[str, Any]]:
        """
        Decide which agents to call based on intent.

        Strategy:
        - refund: ALWAYS call both PolicyExpert + TransactionAgent (parallel)
                  TransactionAgent will handle "no order_id" case gracefully
        - policy: Only PolicyExpert
        - general: PolicyExpert as fallback

        Args:
            intent: Classified intent
            context: Request context (may contain order_id, etc.)

        Returns:
            List of agent call configurations
        """
        calls = []

        if intent == "refund":
            # For refund: ALWAYS need policy + order details
            # Execute in parallel for speed

            # ALWAYS get policy context for refunds
            calls.append({
                "agent": "policy_expert",
                "task": "search_policy",
                "context": {"query": "refund policy requirements"},
                "parallel": True
            })

            # Extract order_id from user message (may be None)
            order_id = self._extract_order_id(context.get("user_message", ""))

            # ALWAYS call TransactionAgent for refund intent
            # If order_id is None, TransactionAgent will handle gracefully
            # (return "not found" which triggers prompt to ask user for order_id)
            calls.append({
                "agent": "transaction_agent",
                "task": "get_order",
                "context": {"order_id": order_id},  # Pass None if not found
                "parallel": True
            })

            if order_id:
                self.logger.info(
                    "order_id_extracted",
                    agent=self.name,
                    order_id=order_id,
                    extraction_method="regex"
                )
            else:
                self.logger.info(
                    "order_id_not_found",
                    agent=self.name,
                    detail="No order_id in user message. TransactionAgent will prompt user."
                )

        elif intent == "policy":
            # For policy questions: Just search policy
            calls.append({
                "agent": "policy_expert",
                "task": "search_policy",
                "context": {"query": context.get("user_message", "")},
                "parallel": False
            })

        elif intent == "general":
            # For general questions: Search policy as fallback
            calls.append({
                "agent": "policy_expert",
                "task": "search_policy",
                "context": {"query": context.get("user_message", "")},
                "parallel": False
            })

        self.logger.info(
            "agent_calls_planned",
            agent=self.name,
            intent=intent,
            num_calls=len(calls),
            agents_to_call=[c["agent"] for c in calls]
        )

        return calls

    def _extract_order_id(self, text: str) -> str | None:
        """
        Extract order ID from text with flexible multilingual pattern matching.

        Handles multiple formats (English + Spanish):
        - "ORD-84315" (full format with prefix)
        - "order 84315", "pedido 25836", "orden 12345" (without prefix)
        - "order number 123456", "pedido número 789", "número de pedido 456"
        - "order #789", "pedido #456"
        - Standalone numbers 4-6 digits when no other context (fallback)

        Args:
            text: User message (English or Spanish)

        Returns:
            Order ID in ORD-XXXXX format if found, else None

        Examples:
            >>> _extract_order_id("quiero devolver mi pedido número 25836")
            'ORD-25836'
            >>> _extract_order_id("I want to return order ORD-84315")
            'ORD-84315'
            >>> _extract_order_id("devolver orden 12345")
            'ORD-12345'
        """
        import re

        # PATTERN 1: Full format with ORD- prefix (highest priority)
        match = re.search(r'ORD-\d{4,6}', text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # PATTERN 2: Number in context of order keywords (English + Spanish)
        # Matches:
        # - English: "order 44012", "order is 44012", "order number 44012", "order #44012"
        # - Spanish: "pedido 25836", "pedido número 25836", "número de pedido 789", "orden 12345"
        order_keywords = r'(?:order|pedido|orden|compra)'
        number_connectors = r'(?:\s+(?:is\s+|number\s+|número\s+|#\s*|n[úu]mero\s+de\s+)?)'

        pattern = rf'\b{order_keywords}{number_connectors}(\d{{4,6}})\b'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            order_number = match.group(1)
            return f"ORD-{order_number}"

        # PATTERN 3: Reverse pattern (número de pedido XXXXX)
        # Matches: "número de pedido 25836", "numero pedido 12345"
        match = re.search(r'\bn[úu]mero\s+(?:de\s+)?(?:pedido|orden)\s+(\d{4,6})\b', text, re.IGNORECASE)
        if match:
            order_number = match.group(1)
            return f"ORD-{order_number}"

        # PATTERN 4: Standalone 4-6 digit number (fallback, lowest priority)
        # Only triggers if no other patterns matched (to avoid false positives)
        match = re.search(r'\b(\d{4,6})\b', text)
        if match:
            order_number = match.group(1)
            self.logger.info(
                "order_id_extracted_fallback",
                agent=self.name,
                order_number=order_number,
                extraction_method="fallback_standalone_number"
            )
            return f"ORD-{order_number}"

        return None

    async def _execute_agent_calls(self, calls: List[Dict]) -> Dict[str, AgentResponse]:
        """
        Execute agent calls (parallel when possible).

        Args:
            calls: List of agent call configurations

        Returns:
            Dict mapping agent names to responses
        """
        results = {}

        # Group by parallel flag
        parallel_calls = [c for c in calls if c.get("parallel", False)]
        sequential_calls = [c for c in calls if not c.get("parallel", False)]

        # Execute parallel calls with asyncio.gather
        if parallel_calls:
            self.logger.info(
                "executing_parallel_calls",
                agent=self.name,
                num_calls=len(parallel_calls)
            )

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
                else:
                    self.logger.warning(
                        "agent_not_found",
                        agent=self.name,
                        requested_agent=call["agent"]
                    )

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for call, response in zip(parallel_calls, responses):
                if isinstance(response, Exception):
                    self.logger.error(
                        "agent_call_exception",
                        agent=self.name,
                        called_agent=call["agent"],
                        error=response
                    )
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
            else:
                self.logger.warning(
                    "agent_not_found",
                    agent=self.name,
                    requested_agent=call["agent"]
                )

        return results

    def _build_context_string(self, results: Dict[str, AgentResponse]) -> str:
        """
        Build context string from agent results.

        Args:
            results: Responses from specialized agents

        Returns:
            Formatted context string for LLM prompt
        """
        context_parts = []
        for agent_name, response in results.items():
            if response.status == "success":
                result_str = str(response.result)
                context_parts.append(f"[{agent_name}]: {result_str}")
            else:
                context_parts.append(f"[{agent_name}]: ERROR - {response.error}")

        return "\n\n".join(context_parts)

    async def _assemble_response(
        self,
        intent: str,
        results: Dict[str, AgentResponse],
        user_message: str
    ) -> Dict[str, Any]:
        """
        Assemble final response using structured outputs.

        Uses AgentResponseTemplate to ensure consistent response format.

        Args:
            intent: Classified intent
            results: Responses from specialized agents
            user_message: Original user query

        Returns:
            Dict with "response" (AgentResponseTemplate) and optionally "eligibility_info" (RefundEligibilityInfo)
        """
        eligibility_info = None

        # Check refund eligibility if we have order data (delegate to TransactionAgent)
        if intent == "refund" and "transaction_agent" in results:
            trans_response = results["transaction_agent"]
            if trans_response.status == "success" and trans_response.result.get("found"):
                order_data = trans_response.result.get("order_data", {})

                # Delegate eligibility check to TransactionAgent
                eligibility_request = AgentRequest(
                    agent="transaction_agent",
                    task="check_eligibility",
                    context={"order_data": order_data}
                )

                eligibility_response = await self.agents["transaction_agent"].handle_request(
                    eligibility_request
                )

                if eligibility_response.status == "success":
                    eligibility_info = RefundEligibilityInfo(**eligibility_response.result)

                    self.logger.info(
                        "refund_eligibility_checked",
                        agent=self.name,
                        eligible=eligibility_info.eligible,
                        days_since_purchase=eligibility_info.days_since_purchase
                    )

        # Build context from agent results
        context_str = self._build_context_string(results)

        # Build eligibility context for prompt (simplified)
        eligibility_context = ""
        if eligibility_info:
            eligibility_context = f"""
Refund Eligibility Check:
- Eligible: {eligibility_info.eligible}
- Reason: {eligibility_info.reason}
- Order Status: {eligibility_info.order_status}
- Days since purchase: {eligibility_info.days_since_purchase or 'N/A'}
"""
            if eligibility_info.already_refunded:
                eligibility_context += f"""- Already refunded: Yes
- Refund date: {eligibility_info.refund_date}
- Transaction ID: {eligibility_info.refund_transaction_id}
"""

        # Load and format prompt from external config
        prompt = get_prompt(
            "response_assembly",
            user_message=user_message,
            intent=intent,
            context_str=context_str,
            eligibility_context=eligibility_context
        )

        config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=AgentResponseTemplate.model_json_schema()
        )

        self.logger.info("assembling_structured_response", agent=self.name)

        try:
            response = await self._call_llm_with_timeout(self.model, prompt, config)
            response_data = AgentResponseTemplate.model_validate_json(response.text)

            self.logger.info(
                "structured_response_assembled",
                agent=self.name,
                response_type=response_data.response_type,
                message_length=len(response_data.message),
                has_action=bool(response_data.action_required)
            )

            result = {"response": response_data}
            if eligibility_info:
                result["eligibility_info"] = eligibility_info

            return result

        except ValidationError as e:
            self.logger.error("structured_response_validation_failed", agent=self.name, error=str(e))

            fallback = AgentResponseTemplate(
                response_type="error",
                message="I apologize, but I encountered an error processing your request. Please contact our support team for assistance.",
                action_required="Contact support@barefootzenith.com"
            )

            return {"response": fallback}
