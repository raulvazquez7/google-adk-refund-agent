# Multi-Agent Refund System with Google ADK

A production-ready **multi-agent system** built with Google's Agent Developer Kit (ADK) that automates customer refund processing using async Agent-to-Agent (A2A) communication, RAG, and LLMs.

> **Demo Use Case**: Customer support automation for "Barefoot Zénit", a premium children's shoe company.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-1.15.1-green)](https://cloud.google.com/adk)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.11-orange)](https://docs.pydantic.dev/)
[![Async](https://img.shields.io/badge/100%25-Async-purple)](https://docs.python.org/3/library/asyncio.html)

---

## 🎯 What It Does

This system **automatically processes refund requests** by:

1. **Classifying user intent** (refund, policy question, general query)
2. **Searching company policies** via semantic search (RAG with Firestore Vector Search)
3. **Retrieving order data** from Firestore database
4. **Validating refund eligibility** (14-day window, order status, etc.)
5. **Processing refunds** with user confirmation
6. **Assembling intelligent responses** using structured LLM outputs

**Result**: Fully automated refund workflow with human-in-the-loop confirmation for safety.

---

## 🏗️ Architecture: Multi-Agent A2A Pattern

Instead of a monolithic agent, this system uses **specialized agents** that collaborate via standardized protocols:

```
┌──────────────────────────────────────────────┐
│       COORDINATOR AGENT (Orchestrator)       │
│  • Classifies intent (LLM)                   │
│  • Routes to specialized agents (parallel)   │
│  • Assembles final response (LLM)            │
└────────────┬─────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌─────▼──────┐
│ POLICY │      │TRANSACTION │
│ EXPERT │      │   AGENT    │
│        │      │            │
│  RAG   │      │ Orders DB  │
│ Search │      │ Refunds    │
│ (async)│      │Eligibility │
└────────┘      └────────────┘
```

### Why Multi-Agent?

- **Scalability**: Easy to add new agents (ShippingAgent, InventoryAgent, etc.)
- **Maintainability**: Each agent is independent and testable
- **Performance**: Parallel execution via `asyncio.gather()` (500ms vs 1000ms sequential)
- **Separation of Concerns**: Policy logic ≠ Transaction logic

---

## 🛠️ Tech Stack

| Component          | Technology                   | Purpose                                |
| ------------------ | ---------------------------- | -------------------------------------- |
| **Agent Framework**| Google ADK 1.15.1            | Multi-agent orchestration              |
| **LLM**            | Gemini 2.5 Flash             | Intent classification & response gen   |
| **Embeddings**     | Vertex AI text-embedding-004 | Vector generation for RAG              |
| **Vector DB**      | Firestore Vector Search      | Semantic search on company policies    |
| **Database**       | Firestore (AsyncClient)      | Order storage & retrieval              |
| **Validation**     | Pydantic 2.11                | Type-safe data contracts               |
| **Observability**  | Langfuse Cloud               | Complete tracing, metrics, costs       |
| **Logging**        | Structured JSON              | Production-ready parseable logs        |
| **Async Runtime**  | asyncio                      | 100% async/await (non-blocking I/O)    |

---

## ✨ Key Features (Production-Ready)

### 🚀 Performance
- **100% Async/Await**: All I/O operations are non-blocking
- **Parallel Agent Execution**: Independent tasks run concurrently (`asyncio.gather`)
- **Rate Limiting**: Shared `Semaphore` to control LLM API saturation
- **Embeddings Cache**: LRU cache reduces redundant API calls (33% hit rate)

### 🔒 Reliability
- **Timeout Protection**: All LLM calls wrapped in `asyncio.wait_for(timeout=30s)`
- **Automatic Retries**: Exponential backoff via `tenacity` (3 attempts)
- **Fail-Fast Validation**: Checks ordered from fastest to slowest
- **Type Safety**: Pydantic models for ALL data contracts

### 📊 Observability
- **Distributed Tracing**: Every interaction logged to Langfuse Cloud
- **Cost Tracking**: Token usage + estimated cost per request
- **Structured Logging**: JSON logs for easy parsing and analysis
- **Performance Metrics**: Latency tracking per agent/tool
- **Context Monitoring**: Real-time token usage stats (e.g., `💾 Context: 12 msgs, 4523 tokens (28.3%)`)

### 🧠 AI/ML Best Practices
- **Structured Outputs**: LLM responses forced into Pydantic schemas (no parsing errors)
- **Externalized Prompts**: Templates in `config/prompts.yaml` with `@lru_cache`
- **RAG Pipeline**: Async embeddings → Vector search → Top-K ranking
- **Agent Protocols**: Standardized `AgentRequest` / `AgentResponse` for A2A communication

### 🧮 Context Engineering (NEW in v1.2.0)
- **Conversation History Management**: Tracks full conversation state across turns
- **Intelligent Compaction**: Automatic summarization when approaching token limits (75% threshold)
- **Token Monitoring**: Real-time token counting with `tiktoken`
- **Sliding Window**: Preserves first + last 8 messages, compresses middle messages
- **LLM Summarization**: Uses Gemini to generate concise 3-4 point summaries (70-80% compression)
- **Fallback Pruning**: Removes less relevant messages if summarization fails
- **Unlimited Conversations**: No context window overflow, supports infinite turns

---

## 🔄 How It Works (Flow)

**User Input**: `"I want to return order ORD-84315"`

```
1. COORDINATOR classifies intent → "refund" (LLM + structured output)

2. COORDINATOR plans agent calls → [PolicyExpert, TransactionAgent] (parallel)

3. AGENTS execute in parallel:
   ├─ PolicyExpert: RAG search on refund policy (async)
   └─ TransactionAgent: Fetch order from Firestore (async)

4. COORDINATOR validates eligibility → TransactionAgent.check_eligibility()
   • Order status: DELIVERED ✅
   • Days since purchase: 12 ✅ (< 14 day window)
   • Not already refunded ✅

5. COORDINATOR assembles response (LLM + structured output)
   → "Order qualifies! 2 days remaining. Reply 'yes' to confirm."

6. USER confirms → TransactionAgent.process_refund()
   → Updates Firestore: status="RETURNED", transaction_id="REF-1727..."
```

**Total latency**: ~1.5s (intent: 300ms, parallel agents: 500ms, eligibility: 200ms, response: 400ms)

---

## 📁 Project Structure

```
.
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Base class: timeout, retries, rate limiting
│   │   ├── coordinator.py          # Orchestrator (intent → routing → assembly)
│   │   ├── policy_expert.py        # RAG specialist
│   │   └── transaction_agent.py    # Order & refund specialist
│   ├── models/
│   │   ├── schemas.py              # Pydantic models (LLM outputs, orders, etc.)
│   │   └── protocols.py            # A2A communication (AgentRequest/Response)
│   ├── utils/
│   │   ├── logger.py               # Structured JSON logging
│   │   ├── prompts.py              # Prompt loading with @lru_cache
│   │   └── conversation_history.py # 🆕 Context engineering & history management
│   ├── tools.py                    # RAG pipeline, Firestore queries
│   └── config.py                   # Centralized settings (Pydantic)
├── config/
│   └── prompts.yaml                # Externalized LLM prompts
├── data/
│   ├── company_refund_policy_barefoot.md  # Policy document (source)
│   └── orders.jsonl                       # Sample orders
├── scripts/
│   ├── 01_seed_orders.py           # Seed Firestore with orders
│   └── 02_setup_vector_search.py   # Generate embeddings & index
├── tests/
│   ├── test_schemas.py             # Pydantic model validation tests
│   ├── test_coordinator_extraction.py  # Order ID extraction tests
│   ├── test_prompts.py             # Prompt loading tests
│   ├── test_system.py              # Automated system tests (3 scenarios)
│   ├── test_refund_flow.py         # End-to-end refund flow test
│   ├── test_multi_agent_system.py  # Multi-agent orchestration tests
│   ├── test_policy_expert.py       # RAG pipeline tests
│   ├── test_transaction_agent.py   # Order & refund tests
│   └── test_async_tools.py         # Async tool tests
├── main_multi_agent.py             # CLI entry point
├── requirements.txt
├── CHANGELOG.md                    # Detailed version history
├── CONTEXT_ENGINEERING_IMPROVEMENTS.md  # 🆕 Context management docs
└── .env.example
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Google Cloud Project** with:
  - Firestore enabled
  - Vertex AI API enabled
- **Langfuse account** (free tier) for observability

### 1. Installation

```bash
git clone <repository-url>
cd ReAct-Google-ADK
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:

```env
# Google Cloud
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1

# AI Models
EMBEDDINGS_MODEL=text-embedding-004
AGENT_MODEL=gemini-2.5-flash

# Langfuse (Observability)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Authenticate & Seed Database

```bash
# Authenticate with GCP
gcloud auth application-default login

# Load sample orders
python scripts/01_seed_orders.py

# Generate embeddings for RAG
python scripts/02_setup_vector_search.py
```

### 4. Run the System

```bash
python main_multi_agent.py
```

---

## 💬 Usage Example

```
🤖 BAREFOOT ZÉNIT - MULTI-AGENT REFUND SYSTEM
Session ID: session_abc123

✨ Features:
  • Automatic refund eligibility checking
  • Confirms before processing refunds
  • Updates order status in database

💬 You: I want to return order ORD-84315

🔄 Processing...

🔍 [Consulted: policy_expert, transaction_agent]

✅ Refund Eligible
──────────────────────────────────────────
Your order qualifies for a refund! You purchased on
September 18, 2025 (12 days ago). Our policy allows
refunds within 14 days of delivery.

📌 Key Details:
  • Order: ORD-84315
  • Status: DELIVERED
  • Days remaining: 2

➡️  Next Step: Reply 'yes' to confirm refund

💡 Reply 'yes' to confirm refund of $89.99 for order ORD-84315

⏱️  Latency: 1523ms
💾 Context: 4 msgs, 1234 tokens (7.7%)
📊 Trace: https://cloud.langfuse.com/project/.../traces/...

💬 You: yes

✅ Refund processed successfully!
   Transaction ID: REF-1727696500000
   Amount: $89.99
   Order ORD-84315 status → RETURNED
   Refund date: 2025-10-24T22:39:11.777542

💾 Context: 6 msgs, 2103 tokens (13.1%)
```

### Long Conversation Example (Context Compaction)

After 12+ messages, the system automatically compresses history:

```json
{
  "message": "compaction_started",
  "total_tokens": 12500
}
{
  "message": "messages_summarized",
  "compression_ratio": "73.2%"
}
{
  "message": "compaction_completed",
  "tokens_after": 8230
}
```

User sees:
```
💾 Context: 9 msgs, 8230 tokens (51.4%)
```

**Result**: Conversation can continue indefinitely without context overflow!

---

## 🔬 Technical Highlights

### 1. Agent Communication Protocol

All agents communicate via standardized Pydantic models:

```python
# Request format (Coordinator → Specialized Agent)
AgentRequest(
    agent="policy_expert",
    task="search_policy",
    context={"query": "refund eligibility"},
    metadata={"session_id": "..."}
)

# Response format (Specialized Agent → Coordinator)
AgentResponse(
    agent="policy_expert",
    status="success",
    result={"policy_text": "..."},
    metadata={"latency_ms": 234, "tokens_used": 120}
)
```

### 2. RAG Pipeline (Async)

```python
async def rag_search_tool(query: str) -> str:
    # 1. Generate embedding (async Vertex AI API)
    query_embedding = await model.get_embeddings_async([query])

    # 2. Retrieve chunks from Firestore (async)
    chunks = await retrieve_policy_chunks_async()

    # 3. Rank by cosine similarity (sync, CPU-bound)
    top_results = rank_by_similarity(query_embedding, chunks, top_k=3)

    # 4. Return concatenated text
    return "\n---\n".join([r["text"] for r in top_results])
```

### 3. Structured LLM Outputs (Pydantic)

Force LLM to return valid JSON matching Pydantic schema:

```python
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: Literal["refund", "policy", "general"]
    confidence: float = Field(ge=0.0, le=1.0)

# LLM call with schema enforcement
config = GenerationConfig(
    response_mime_type="application/json",
    response_schema=IntentClassification.model_json_schema()
)

response = await model.generate_content_async(prompt, generation_config=config)
intent = IntentClassification.model_validate_json(response.text)  # ✅ Type-safe
```

### 4. Parallel Execution

```python
# ❌ Sequential (slow)
policy_result = await policy_expert.handle_request(...)  # 500ms
order_result = await transaction_agent.handle_request(...)  # 500ms
# Total: 1000ms

# ✅ Parallel (fast)
results = await asyncio.gather(
    policy_expert.handle_request(...),
    transaction_agent.handle_request(...)
)
# Total: 500ms (max of both)
```

### 5. Conversation History Management (Context Engineering)

**Intelligent context window management for unlimited conversations:**

```python
from src.utils.conversation_history import ConversationHistoryManager

# Initialize with context engineering settings
history_manager = ConversationHistoryManager(
    max_tokens=16000,         # Gemini 2.0 Flash context window
    target_tokens=12000,      # Trigger compaction at 75%
    keep_recent_messages=8,   # Always preserve last 8 messages
    enable_summarization=True # Auto-summarize old messages
)

# Track conversation
history_manager.add_message("user", "What is your refund policy?")
history_manager.add_message("assistant", "Our refund policy allows...")

# Get stats
stats = history_manager.get_stats()
# {'total_messages': 12, 'total_tokens': 4523, 'token_usage_percent': 28.3}

# Get formatted context for LLM
context = history_manager.get_context_for_llm()
```

**Compaction Strategy (Sliding Window)**:
```
┌─────────────────────────────────────┐
│ First Message (system context)     │ ← Always preserved
├─────────────────────────────────────┤
│ Middle Messages (10 msgs, 5K tok)  │
│     ↓ Summarization (Gemini)       │
│ Summary (3-4 bullets, 1K tokens)   │ ← 80% compression
├─────────────────────────────────────┤
│ Recent 8 Messages                   │ ← Always preserved
└─────────────────────────────────────┘
```

**Benefits**:
- ✅ Unlimited conversation length (no context overflow)
- ✅ 70-80% token compression via LLM summarization
- ✅ Real-time monitoring: `💾 Context: 12 msgs, 4523 tokens (28.3%)`
- ✅ Automatic fallback to pruning if summarization fails

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests with pytest
pytest tests/ -v

# Run specific test modules
python tests/test_system.py              # 3 automated system tests
python tests/test_refund_flow.py         # End-to-end refund flow
python tests/test_multi_agent_system.py  # Multi-agent orchestration
```

**Test Coverage**:
- ✅ Unit tests for each agent (`test_policy_expert.py`, `test_transaction_agent.py`)
- ✅ Integration tests for multi-agent orchestration (`test_multi_agent_system.py`)
- ✅ RAG pipeline tests (`test_async_tools.py`)
- ✅ Pydantic schema validation tests (`test_schemas.py`)
- ✅ Order ID extraction tests (Spanish + English) (`test_coordinator_extraction.py`)
- ✅ Prompt loading tests (`test_prompts.py`)
- ✅ End-to-end refund flow tests (`test_refund_flow.py`)

**Sample Test Scenarios**:
- ✅ Eligible refund (within 14 days, DELIVERED status)
- ✅ Ineligible refund (>14 days)
- ✅ Already refunded order
- ✅ Policy-only queries
- ✅ Spanish input without ORD- prefix (`"pedido número 25836"` → `"ORD-25836"`)
- ✅ General queries (out of scope)

---

## 📊 Design Patterns Implemented

- **Template Method**: `BaseAgent` defines skeleton, subclasses implement `_execute_task()`
- **Factory**: `AgentResponse.create_success()` / `create_error()` helpers
- **Strategy**: Coordinator routes to different agents based on intent
- **Singleton**: Global `settings` instance, logger cache
- **Fail-Fast**: Eligibility checks ordered from fastest to slowest

---

## 🎯 Future Enhancements

- [x] ~~Embeddings cache (LRU) for repeated queries~~ ✅ **Implemented** (33% hit rate)
- [x] ~~Conversation history management~~ ✅ **Implemented** (v1.2.0)
- [x] ~~Context window compaction~~ ✅ **Implemented** (70-80% compression)
- [ ] Pass conversation history to Coordinator for context-aware responses
- [ ] Persist conversation history to Firestore for multi-session continuity
- [ ] Circuit breaker for Firestore failures
- [ ] Human-in-the-loop for high-value refunds (>$500)
- [ ] A/B testing for prompt variations
- [ ] Multi-language support (i18n)
- [ ] Streaming responses for better UX
- [ ] Semantic search over conversation history (embeddings + vector DB)

---

## 📚 References

- [Google ADK Documentation](https://cloud.google.com/adk)
- [Firestore Vector Search](https://cloud.google.com/firestore/docs/vector-search)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [Langfuse Observability](https://langfuse.com/docs)
- [Context Engineering Improvements](./CONTEXT_ENGINEERING_IMPROVEMENTS.md) - Deep dive into v1.2.0 changes
- [Full Changelog](./CHANGELOG.md) - Complete version history

---

## 📝 License

MIT License - This is a portfolio/learning project demonstrating production-ready AI agent architecture.

---

## 👤 Author

Built as a technical demonstration of:
- Multi-agent systems (A2A communication)
- Production-ready async Python architecture
- RAG implementation with vector databases
- LLM integration best practices
- Cloud-native AI applications
- Context engineering & conversation memory management

**Version**: 1.2.0
**Last Updated**: October 24, 2025

### Key Milestones
- **v1.0.0** (Oct 20, 2025): Initial multi-agent system with RAG
- **v1.1.0** (Oct 24, 2025): Enhanced order ID extraction (multilingual support)
- **v1.2.0** (Oct 24, 2025): **Context engineering & conversation history management**

For questions or collaboration: [Contact Info]
