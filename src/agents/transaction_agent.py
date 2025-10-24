"""
TransactionAgent - Specialist in order management and refund processing.

This agent handles all transaction-related operations including order retrieval
and refund execution.
"""
from typing import Any, Dict

from langfuse import Langfuse

from src.agents.base_agent import BaseAgent
from src.models.protocols import AgentRequest
from src.tools import get_order_details, process_refund


class TransactionAgent(BaseAgent):
    """
    Specialized agent for transaction operations.

    This agent:
    - Retrieves order details from Firestore
    - Processes refunds for eligible orders
    - Validates order existence and data integrity

    Supported tasks:
        - "get_order": Retrieve order details by order_id
        - "process_refund": Execute refund for an order
        - "check_eligibility": Validate if order qualifies for refund

    Example:
        agent = TransactionAgent(tracer=langfuse)

        # Get order
        request = AgentRequest(
            agent="transaction_agent",
            task="get_order",
            context={"order_id": "ORD-84315"}
        )
        response = await agent.handle_request(request)

        # Process refund
        request = AgentRequest(
            agent="transaction_agent",
            task="process_refund",
            context={"order_id": "ORD-84315", "amount": 89.99}
        )
        response = await agent.handle_request(request)
    """

    def __init__(self, tracer: Langfuse):
        """
        Initialize the TransactionAgent.

        Args:
            tracer: Langfuse client for observability
        """
        super().__init__(name="transaction_agent", tracer=tracer)

    async def _execute_task(self, request: AgentRequest) -> Any:
        """
        Execute transaction-related tasks.

        Args:
            request: Agent request with task and context

        Returns:
            Task result (dict with order/refund information)

        Raises:
            ValueError: If task is not supported
        """
        task = request.task
        context = request.context

        if task == "get_order":
            return await self._get_order(context)
        elif task == "process_refund":
            return await self._process_refund(context)
        elif task == "check_eligibility":
            return await self._check_eligibility(context)
        else:
            raise ValueError(
                f"Unsupported task: {task}. "
                f"TransactionAgent supports: 'get_order', 'process_refund', 'check_eligibility'"
            )

    async def _get_order(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve order details from Firestore asynchronously.

        Now uses Pydantic OrderResponse instead of JSON parsing.

        Handles the case when order_id is None (user didn't provide it):
        - Returns found=False with a specific error message
        - Coordinator will use this to prompt user for order_id

        Args:
            context: Dictionary containing:
                - order_id (str | None): Order identifier (e.g., "ORD-84315") or None

        Returns:
            Dictionary with:
                - order_id (str | None): The order ID (or None)
                - order_data (OrderData): Pydantic model if found
                - found (bool): Whether order was found
                - error (str): Error message if not found

        Raises:
            Never raises - errors are returned in the dict
        """
        order_id = context.get("order_id")

        # Handle case when user didn't provide order_id
        if not order_id:
            self.logger.info(
                "get_order_no_id_provided",
                agent=self.name,
                message="No order_id provided by user"
            )
            return {
                "order_id": None,
                "order_data": None,
                "found": False,
                "error": "MISSING_ORDER_ID",  # Special flag for coordinator
                "user_message": "Por favor, proporciona tu número de pedido para procesar la devolución."
            }

        self.logger.info(
            "get_order_started",
            agent=self.name,
            order_id=order_id
        )

        # Call async tool from src.tools (returns OrderResponse Pydantic)
        result = await get_order_details(order_id)

        if result.found:
            self.logger.info(
                "get_order_completed",
                agent=self.name,
                order_id=order_id,
                found=True,
                order_status=result.order_data.status
            )

            return {
                "order_id": order_id,
                "order_data": result.order_data.model_dump(),
                "found": True
            }
        else:
            self.logger.warning(
                "get_order_not_found",
                agent=self.name,
                order_id=order_id
            )

            return {
                "order_id": order_id,
                "order_data": None,
                "found": False,
                "error": result.error
            }

    async def _process_refund(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a refund for an order asynchronously.

        Now uses Pydantic RefundProcessingResult instead of JSON parsing.

        Args:
            context: Dictionary containing:
                - order_id (str): Order to refund
                - amount (float): Refund amount

        Returns:
            Dictionary with:
                - order_id (str): The order ID
                - amount (float): Refund amount
                - transaction_id (str): Refund transaction ID
                - success (bool): Whether refund succeeded

        Raises:
            ValueError: If order_id or amount is missing
        """
        order_id = context.get("order_id")
        amount = context.get("amount")

        if not order_id:
            raise ValueError("Missing required field: 'order_id' in context")
        if amount is None:
            raise ValueError("Missing required field: 'amount' in context")

        # Validate amount is positive
        try:
            amount = float(amount)
            if amount <= 0:
                raise ValueError(f"Amount must be positive, got: {amount}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid amount value: {amount}") from e

        self.logger.info(
            "process_refund_started",
            agent=self.name,
            order_id=order_id,
            amount=amount
        )

        # Call async tool from src.tools (returns RefundProcessingResult Pydantic)
        result = await process_refund(order_id, amount)

        if result.success:
            self.logger.info(
                "process_refund_completed",
                agent=self.name,
                order_id=order_id,
                amount=amount,
                transaction_id=result.transaction_id
            )

            return {
                "order_id": order_id,
                "amount": amount,
                "transaction_id": result.transaction_id,
                "refund_date": result.refund_date,
                "success": True
            }
        else:
            self.logger.error(
                "process_refund_failed",
                agent=self.name,
                order_id=order_id,
                amount=amount,
                error=result.error
            )

            return {
                "order_id": order_id,
                "amount": amount,
                "success": False,
                "error": result.error
            }

    async def _check_eligibility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an order is eligible for refund.

        Moved from Coordinator for separation of concerns.
        Transaction-related business logic belongs in TransactionAgent.

        Validation order (fail-fast):
        1. Already returned? → No refund
        2. Status != DELIVERED? → Cannot refund
        3. Purchase date > 14 days? → Outside window
        4. All pass → Eligible

        Args:
            context: Dictionary containing:
                - order_data (dict): Order information

        Returns:
            RefundEligibilityInfo as dict

        Raises:
            ValueError: If order_data is missing
        """
        from datetime import datetime
        from src.models.schemas import RefundEligibilityInfo

        order_data = context.get("order_data")
        if not order_data:
            raise ValueError("Missing required field: 'order_data' in context")

        order_status = order_data.get("status", "").upper()

        # STEP 1: Check if already refunded
        if order_status == "RETURNED":
            refund_date = order_data.get("refund_date")
            refund_transaction_id = order_data.get("refund_transaction_id")
            refund_amount = order_data.get("refund_amount")

            self.logger.info(
                "eligibility_check_already_refunded",
                agent=self.name,
                order_status=order_status,
                refund_date=refund_date
            )

            eligibility = RefundEligibilityInfo(
                eligible=False,
                already_refunded=True,
                invalid_status=False,
                order_status=order_status,
                reason="Order was already refunded",
                refund_transaction_id=refund_transaction_id,
                refund_date=refund_date,
                refund_amount=refund_amount
            )
            return eligibility.model_dump()

        # STEP 2: Check if order is DELIVERED (only delivered orders can be refunded)
        if order_status != "DELIVERED":
            self.logger.info(
                "eligibility_check_invalid_status",
                agent=self.name,
                order_status=order_status,
                required_status="DELIVERED"
            )

            eligibility = RefundEligibilityInfo(
                eligible=False,
                already_refunded=False,
                invalid_status=True,
                order_status=order_status,
                reason=f"Order status is '{order_status}'. Only DELIVERED orders can be refunded."
            )
            return eligibility.model_dump()

        # STEP 3: Check purchase date eligibility
        purchase_date_str = order_data.get("purchase_date")
        if not purchase_date_str:
            eligibility = RefundEligibilityInfo(
                eligible=False,
                already_refunded=False,
                invalid_status=False,
                order_status=order_status,
                reason="Purchase date not found in order data"
            )
            return eligibility.model_dump()

        try:
            if isinstance(purchase_date_str, str):
                purchase_date = datetime.fromisoformat(purchase_date_str.replace('Z', '+00:00'))
            else:
                purchase_date = purchase_date_str

            now = datetime.now(purchase_date.tzinfo) if purchase_date.tzinfo else datetime.now()
            days_elapsed = (now - purchase_date).days

            REFUND_WINDOW_DAYS = 14

            if days_elapsed <= REFUND_WINDOW_DAYS:
                eligibility = RefundEligibilityInfo(
                    eligible=True,
                    already_refunded=False,
                    invalid_status=False,
                    order_status=order_status,
                    days_since_purchase=days_elapsed,
                    days_remaining=REFUND_WINDOW_DAYS - days_elapsed,
                    reason=f"Order is within {REFUND_WINDOW_DAYS}-day refund window"
                )
                return eligibility.model_dump()
            else:
                eligibility = RefundEligibilityInfo(
                    eligible=False,
                    already_refunded=False,
                    invalid_status=False,
                    order_status=order_status,
                    days_since_purchase=days_elapsed,
                    reason=f"Order is {days_elapsed} days old, exceeds {REFUND_WINDOW_DAYS}-day limit"
                )
                return eligibility.model_dump()

        except Exception as e:
            self.logger.error("eligibility_check_failed", error=e)
            eligibility = RefundEligibilityInfo(
                eligible=False,
                already_refunded=False,
                invalid_status=False,
                order_status=order_status,
                reason=f"Error checking eligibility: {str(e)}"
            )
            return eligibility.model_dump()
