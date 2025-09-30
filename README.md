# Barefoot Zénit Refund Agent

A production-ready ReAct agent built with Google ADK for handling customer refund requests. This project demonstrates best practices for building agentic systems with LLMs, RAG, and cloud-native infrastructure.

## 🎯 Project Overview

This agent assists customer support representatives at "Barefoot Zénit", a premium children's shoe company, by:
- Answering policy questions using semantic search (RAG)
- Retrieving order details from a database
- Making intelligent refund decisions based on company policies
- Processing refunds automatically when eligible

**Status:** 🚧 In active development

## 🏗️ Architecture

The agent follows the **ReAct (Reason + Act) pattern**, where the LLM:
1. **Reasons** about what information it needs
2. **Acts** by calling tools to gather that information
3. **Repeats** until it has enough context to provide a final answer

### Tech Stack

- **Agent Framework:** Google Agent Developer Kit (ADK)
- **LLM:** Gemini 2.5 Flash (configurable)
- **Vector Database:** Firestore Vector Search
- **Embeddings:** Vertex AI Text Embedding (text-embedding-004)
- **Observability:** Langfuse Cloud
- **Infrastructure:** Google Cloud Platform (Firestore, Vertex AI)

## 🧠 Key Design Decisions

### Why Firestore Vector Search over Matching Engine?

This project initially explored **Vertex AI Vector Search (Matching Engine)** but ultimately chose **Firestore Vector Search** for the following reasons:

| Aspect | Matching Engine | Firestore Vector Search | Our Choice |
|--------|----------------|------------------------|------------|
| **Setup Time** | 30-60 minutes per index | < 1 minute | ✅ Firestore |
| **Best For** | Millions+ vectors | Small-medium datasets | ✅ Firestore |
| **Complexity** | High (requires GCS, index management) | Low (native Firestore) | ✅ Firestore |
| **Cost** | Higher (dedicated infra) | Lower (pay-per-use) | ✅ Firestore |
| **Our Use Case** | ~10 policy chunks | ~10 policy chunks | ✅ Firestore |

**Conclusion:** For datasets under 10,000 vectors, Firestore Vector Search provides excellent performance with dramatically simpler setup. The Matching Engine exploration scripts are preserved in `scripts/test_matching_engine/` for reference.

### Why Dynamic Date Injection?

The agent's system prompt includes the current date/time injected dynamically at runtime. This ensures:
- Accurate refund eligibility calculations (e.g., "14 days from purchase")
- No stale date information
- Production-ready temporal awareness

## 📁 Project Structure

```
.
├── data/
│   ├── company_refund_policy_barefoot.md  # Source policy document
│   └── orders.jsonl                       # Sample order data
├── scripts/
│   ├── 01_seed_orders.py                  # Loads orders into Firestore
│   ├── 02_setup_vector_search.py          # Generates and stores embeddings
│   └── test_matching_engine/              # Matching Engine exploration (archived)
├── src/
│   ├── agent.py                           # Agent definition & system prompt
│   └── tools.py                           # Tool implementations (RAG, Firestore)
├── main.py                                # CLI entry point
├── requirements.txt
└── .env.example                           # Environment variables template
```

## 🚀 Setup

### Prerequisites

- Python 3.10+
- Google Cloud Project with:
  - Firestore enabled
  - Vertex AI API enabled
- Langfuse account (free tier)

### 1. Clone & Install

```bash
git clone <repository-url>
cd "ReAct Google ADK"
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Google Cloud Configuration
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
EMBEDDINGS_MODEL=text-embedding-004
AGENT_MODEL=gemini-2.5-flash

# Langfuse Configuration (for observability)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Authenticate with GCP

```bash
gcloud auth application-default login
```

### 4. Seed the Database

```bash
# Load sample orders into Firestore
python scripts/01_seed_orders.py

# Generate embeddings and set up vector search
python scripts/02_setup_vector_search.py
```

## 💬 Usage

Run the agent:

```bash
python main.py
```

**Example interaction:**

```
You: Hello, I want to return my order ORD-84315. Does it qualify for a refund?

Agent: Let me check your order details and our refund policy...
[Agent calls rag_search_tool and get_order_details]

Agent: Your order ORD-84315 was placed on September 18, 2025. Today is September 30, 2025,
which means 12 days have passed. According to our policy, refunds are accepted within 14
days of purchase. Your order qualifies!

I've processed the refund successfully. Transaction ID: REF-1234567890

🔍 View trace: https://cloud.langfuse.com/project/.../traces/...
```

## 🔍 Observability

Every interaction is traced in **Langfuse Cloud**, providing:
- Complete conversation history
- Tool call timings and success rates
- LLM token usage and costs
- Error tracking and debugging

Click the trace URL after each interaction to inspect the agent's reasoning process.

## 🛠️ Tools Available to the Agent

| Tool | Purpose | Implementation |
|------|---------|----------------|
| `rag_search_tool` | Semantic search on refund policy | Firestore Vector Search |
| `get_order_details` | Fetch order info by ID | Firestore query |
| `process_refund` | Execute refund (simulated) | Mock payment gateway |

## 🧪 Testing

**Sample Order IDs:**
- `ORD-84315` - Eligible for refund (recent purchase)
- `ORD-84316` - Test different scenarios

**Test Queries:**
- "Can I return shoes if I've only tried them indoors?"
- "What's your refund policy for used items?"
- "I want to return order ORD-84315"

## 📚 Learning Resources

This project was built following:
- [Google ADK Documentation](https://cloud.google.com/adk)
- [Firestore Vector Search Guide](https://cloud.google.com/firestore/docs/vector-search)
- [ReAct Paper: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

## 🚧 Roadmap

- [ ] Add conversation memory/history
- [ ] Implement human-in-the-loop for edge cases
- [ ] Add more comprehensive test suite
- [ ] Deploy to Cloud Run
- [ ] Add streaming responses
- [ ] Multi-language support

## 📝 License

This is a learning/portfolio project. Feel free to fork and experiment!
