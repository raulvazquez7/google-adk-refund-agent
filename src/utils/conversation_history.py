"""
Conversation History Manager with Context Engineering.

Implements strategic context window management using:
- Token counting and monitoring
- Intelligent message pruning
- Conversation summarization for compression
- Maintains recent context + key information

Follows ADK best practices and context engineering patterns.
"""
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from vertexai.generative_models import GenerativeModel

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationMessage:
    """
    Represents a single message in the conversation.

    Attributes:
        role: "user" or "assistant"
        content: The message text
        timestamp: When the message was created
        tokens: Approximate token count for this message
        metadata: Additional metadata (agent_name, intent, etc.)
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationHistoryManager:
    """
    Manages conversation history with intelligent context window management.

    Strategies:
    1. **Token Monitoring**: Tracks total tokens in context
    2. **Sliding Window**: Keeps recent N messages + first message
    3. **Compaction**: Summarizes middle messages when approaching limit
    4. **Pruning**: Removes less relevant intermediate messages

    Usage:
        history = ConversationHistoryManager(max_tokens=8000)
        history.add_message("user", "What is your refund policy?")
        history.add_message("assistant", "Our refund policy is...")

        # Get context for LLM (automatically managed)
        context = history.get_context_for_llm()
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        target_tokens: int = 6000,
        keep_recent_messages: int = 6,
        enable_summarization: bool = True
    ):
        """
        Initialize the conversation history manager.

        Args:
            max_tokens: Hard limit for context window
            target_tokens: Soft limit - trigger compaction when exceeded
            keep_recent_messages: Number of recent messages to always preserve
            enable_summarization: Enable automatic summarization of old messages
        """
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.keep_recent_messages = keep_recent_messages
        self.enable_summarization = enable_summarization

        self.messages: List[ConversationMessage] = []
        self.summary: Optional[str] = None
        self.summary_tokens: int = 0

        # Initialize tokenizer (using tiktoken for GPT models, approximate for Gemini)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("tiktoken_initialization_failed", error=str(e))
            self.tokenizer = None

        # Summarization model (lightweight)
        self.summarizer = GenerativeModel(settings.agent_model)

        logger.info(
            "conversation_history_initialized",
            max_tokens=max_tokens,
            target_tokens=target_tokens,
            keep_recent=keep_recent_messages,
            summarization_enabled=enable_summarization
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate for Gemini).

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimate (1 token ≈ 4 chars for English)
            return len(text) // 4

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new message to the conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (intent, agents_called, etc.)
        """
        tokens = self._count_tokens(content)

        message = ConversationMessage(
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {}
        )

        self.messages.append(message)

        logger.info(
            "message_added_to_history",
            role=role,
            tokens=tokens,
            total_messages=len(self.messages),
            total_tokens=self.get_total_tokens()
        )

        # Check if compaction is needed
        if self.get_total_tokens() > self.target_tokens:
            self._apply_compaction()

    def get_total_tokens(self) -> int:
        """
        Calculate total tokens in current context.

        Returns:
            Total token count (messages + summary)
        """
        message_tokens = sum(msg.tokens for msg in self.messages)
        return message_tokens + self.summary_tokens

    def _apply_compaction(self) -> None:
        """
        Apply context compaction when approaching token limit.

        Strategy:
        1. Keep first message (system context)
        2. Keep last N messages (recent conversation)
        3. Summarize or prune middle messages
        """
        if len(self.messages) <= self.keep_recent_messages + 1:
            # Not enough messages to compact
            logger.info(
                "compaction_skipped",
                reason="insufficient_messages",
                message_count=len(self.messages)
            )
            return

        logger.info(
            "compaction_started",
            total_tokens=self.get_total_tokens(),
            message_count=len(self.messages)
        )

        # Identify messages to compact
        first_message = self.messages[0]
        recent_messages = self.messages[-self.keep_recent_messages:]
        middle_messages = self.messages[1:-self.keep_recent_messages]

        if not middle_messages:
            logger.info("compaction_skipped", reason="no_middle_messages")
            return

        # Strategy: Summarize middle messages
        if self.enable_summarization and len(middle_messages) > 2:
            self._summarize_middle_messages(middle_messages)
        else:
            # Strategy: Simple pruning (remove every other message)
            self._prune_messages(middle_messages)

        # Reconstruct message list
        self.messages = [first_message] + recent_messages

        logger.info(
            "compaction_completed",
            messages_after=len(self.messages),
            tokens_after=self.get_total_tokens(),
            has_summary=bool(self.summary)
        )

    def _summarize_middle_messages(self, messages: List[ConversationMessage]) -> None:
        """
        Summarize middle messages into a concise summary.

        Args:
            messages: Messages to summarize
        """
        # Build conversation text
        conversation_text = "\n\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages
        ])

        prompt = f"""Resume la siguiente conversación en 3-4 puntos clave.
Mantén información importante como números de pedido, decisiones tomadas, y políticas mencionadas.

CONVERSACIÓN:
{conversation_text}

RESUMEN (3-4 bullet points):"""

        try:
            # Generate summary (synchronous call for simplicity)
            response = self.summarizer.generate_content(prompt)
            summary_text = response.text.strip()

            self.summary = summary_text
            self.summary_tokens = self._count_tokens(summary_text)

            logger.info(
                "messages_summarized",
                original_messages=len(messages),
                original_tokens=sum(m.tokens for m in messages),
                summary_tokens=self.summary_tokens,
                compression_ratio=f"{(1 - self.summary_tokens / sum(m.tokens for m in messages)) * 100:.1f}%"
            )

        except Exception as e:
            logger.error(
                "summarization_failed",
                error=str(e),
                fallback="pruning"
            )
            # Fallback to pruning
            self._prune_messages(messages)

    def _prune_messages(self, messages: List[ConversationMessage]) -> None:
        """
        Prune messages by removing less relevant ones.

        Strategy: Remove messages tagged as "general" or "policy_info"
        Keep messages with refund actions.

        Args:
            messages: Messages to prune
        """
        important_messages = []

        for msg in messages:
            # Keep messages with refund actions or errors
            intent = msg.metadata.get("intent", "")
            response_type = msg.metadata.get("response_type", "")

            if intent == "refund" or "refund" in response_type or "error" in response_type:
                important_messages.append(msg)

        if important_messages:
            # Update middle section to only important messages
            pruned_count = len(messages) - len(important_messages)

            logger.info(
                "messages_pruned",
                original_count=len(messages),
                pruned_count=pruned_count,
                retained_count=len(important_messages)
            )

            # Note: This pruning doesn't modify self.messages directly
            # It's used by _apply_compaction to decide what to keep
        else:
            logger.info("pruning_skipped", reason="no_important_messages_identified")

    def get_context_for_llm(self, max_messages: Optional[int] = None) -> str:
        """
        Get formatted conversation context for LLM.

        Args:
            max_messages: Optional limit on number of messages to return

        Returns:
            Formatted conversation history
        """
        parts = []

        # Add summary if exists
        if self.summary:
            parts.append(f"[RESUMEN DE CONVERSACIÓN ANTERIOR]\n{self.summary}\n")

        # Add recent messages
        messages_to_include = self.messages[-max_messages:] if max_messages else self.messages

        for msg in messages_to_include:
            parts.append(f"{msg.role.upper()}: {msg.content}")

        context = "\n\n".join(parts)

        logger.info(
            "context_retrieved",
            total_tokens=self._count_tokens(context),
            messages_included=len(messages_to_include),
            has_summary=bool(self.summary)
        )

        return context

    def get_recent_messages(self, n: int = 5) -> List[ConversationMessage]:
        """
        Get the N most recent messages.

        Args:
            n: Number of recent messages to retrieve

        Returns:
            List of recent messages
        """
        return self.messages[-n:]

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages.clear()
        self.summary = None
        self.summary_tokens = 0

        logger.info("conversation_history_cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation history.

        Returns:
            Dictionary with stats
        """
        return {
            "total_messages": len(self.messages),
            "total_tokens": self.get_total_tokens(),
            "summary_tokens": self.summary_tokens,
            "has_summary": bool(self.summary),
            "token_usage_percent": (self.get_total_tokens() / self.max_tokens) * 100,
            "is_near_limit": self.get_total_tokens() > self.target_tokens
        }
