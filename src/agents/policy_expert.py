"""
PolicyExpert Agent - Specialist in company policy search using RAG.

This agent is responsible for semantic search on the company's refund policy
documents using Retrieval-Augmented Generation (RAG).
"""
from typing import Any, Dict

from langfuse import Langfuse

from src.agents.base_agent import BaseAgent
from src.models.protocols import AgentRequest
from src.tools import rag_search_tool


class PolicyExpertAgent(BaseAgent):
    """
    Specialized agent for policy document search using RAG.

    This agent:
    - Performs semantic search on company policies
    - Returns relevant chunks with similarity scores
    - Handles policy-related queries efficiently

    Supported tasks:
        - "search_policy": Search for relevant policy information

    Example:
        agent = PolicyExpertAgent(name="policy_expert", tracer=langfuse)
        request = AgentRequest(
            agent="policy_expert",
            task="search_policy",
            context={"query": "Can I return shoes after 14 days?"}
        )
        response = await agent.handle_request(request)
    """

    def __init__(self, tracer: Langfuse):
        """
        Initialize the PolicyExpert agent.

        Args:
            tracer: Langfuse client for observability
        """
        super().__init__(name="policy_expert", tracer=tracer)

    async def _execute_task(self, request: AgentRequest) -> Any:
        """
        Execute policy-related tasks.

        Args:
            request: Agent request with task and context

        Returns:
            Task result (dict with policy information)

        Raises:
            ValueError: If task is not supported
            RuntimeError: If RAG search fails
        """
        task = request.task
        context = request.context

        if task == "search_policy":
            return await self._search_policy(context)
        else:
            raise ValueError(f"Unsupported task: {task}. PolicyExpert only supports 'search_policy'")

    async def _search_policy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search company policy using RAG.

        Args:
            context: Dictionary containing:
                - query (str): User's search query

        Returns:
            Dictionary with:
                - policy_text (str): Relevant policy sections
                - query (str): Original query
                - source (str): "RAG search on refund policy"

        Raises:
            ValueError: If query is missing
            RuntimeError: If RAG search fails
        """
        query = context.get("query")

        if not query:
            raise ValueError("Missing required field: 'query' in context")

        self.logger.info(
            "policy_search_started",
            agent=self.name,
            query=query
        )

        # Call the async RAG tool from src.tools
        try:
            policy_text = await rag_search_tool(query)

            self.logger.info(
                "policy_search_completed",
                agent=self.name,
                query=query,
                result_length=len(policy_text)
            )

            return {
                "policy_text": policy_text,
                "query": query,
                "source": "RAG search on refund policy"
            }

        except Exception as e:
            self.logger.error(
                "policy_search_failed",
                error=e,
                agent=self.name,
                query=query
            )
            raise RuntimeError(f"RAG search failed: {str(e)}") from e
