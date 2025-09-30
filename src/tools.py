"""
Tool implementations for the Barefoot Zénit refund agent.

This module provides the concrete implementations of the tools that the agent
can use to perform its tasks. These tools interact with Firestore and Vertex AI
to gather information and execute actions.

Available tools:
- rag_search_tool: Performs semantic search on the refund policy.
- get_order_details: Retrieves specific order information from Firestore.
- process_refund: Simulates processing a refund and returns a transaction ID.
"""
import json
import logging
import os
from datetime import datetime
import numpy as np

from google.cloud import firestore
from vertexai.language_models import TextEmbeddingModel

# --- Configuration ---
# Load environment variables to access GCP resources
# and the embeddings model name.
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
ME_ENDPOINT_NAME = os.getenv("ME_ENDPOINT_NAME")
ME_DEPLOYED_INDEX_ID = os.getenv("ME_DEPLOYED_INDEX_ID")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
FIRESTORE_DATABASE_ID = "orders"  # Use the same database for everything

# Initialize clients
# aiplatform.init(project=PROJECT_ID, location=LOCATION) # This line is no longer needed
db = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DATABASE_ID)


def cosine_similarity(a, b):
    """Calculates cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rag_search_tool(query: str) -> str:
    """
    Performs semantic search on the company's refund policy using Firestore.
    
    This is a production-ready approach for small-to-medium datasets.
    For millions of vectors, consider Vertex AI Vector Search.
    """
    logging.info(f"Performing RAG search with query: '{query}'")
    try:
        # 1. Generate query embedding
        model = TextEmbeddingModel.from_pretrained(EMBEDDINGS_MODEL)
        query_embeddings = model.get_embeddings([query])
        query_vector = np.array(query_embeddings[0].values)
        
        # 2. Retrieve all chunks from Firestore
        collection_ref = db.collection("policy_chunks")
        docs = collection_ref.stream()
        
        # 3. Calculate similarity with each chunk
        results = []
        for doc in docs:
            data = doc.to_dict()
            chunk_vector = np.array(data["embedding"])
            similarity = cosine_similarity(query_vector, chunk_vector)
            results.append({
                "text": data["text"],
                "similarity": similarity,
                "chunk_id": data["chunk_id"]
            })
        
        # 4. Sort by similarity and take top 3
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:3]
        
        if not top_results:
            return "No relevant information found in the refund policy."
        
        # 5. Return the most relevant chunks
        context_pieces = [r["text"] for r in top_results]
        logging.info(f"RAG search found {len(context_pieces)} relevant chunks with similarities: {[f'{r['similarity']:.3f}' for r in top_results]}")
        
        return "\n---\n".join(context_pieces)
        
    except Exception as e:
        logging.error(f"Error during RAG search: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return "An error occurred while querying the refund policy."


def get_order_details(order_id: str) -> str:
    """
    Retrieves the details for a specific order from the Firestore database.
    
    Args:
        order_id: The unique identifier for the order (e.g., "ORD-12345").
        
    Returns:
        A JSON string with the order details or an error message.
    """
    logging.info(f"Fetching details for order from Firestore: {order_id}")
    try:
        doc_ref = db.collection("orders").document(order_id)
        doc = doc_ref.get()

        if not doc.exists:
            logging.warning(f"Order '{order_id}' not found in Firestore.")
            return json.dumps({"status": "error", "message": "Order not found."})

        order_data = doc.to_dict()
        
        # Firestore returns datetime objects, we convert them to strings for the LLM
        if 'purchase_date' in order_data and isinstance(order_data['purchase_date'], datetime):
            order_data['purchase_date'] = order_data['purchase_date'].isoformat()

        return json.dumps({"status": "success", "data": order_data})

    except Exception as e:
        logging.error(f"Error fetching order '{order_id}' from Firestore: {e}")
        return json.dumps({"status": "error", "message": "An error occurred while fetching order details."})


def process_refund(order_id: str, amount: float) -> str:
    """
    Processes a refund for a given order and amount.
    
    In a real system, this would call a payment gateway API (e.g., Stripe).
    Here, we just simulate a successful response.
    
    Args:
        order_id: The order ID to be refunded.
        amount: The amount to refund.
        
    Returns:
        A JSON string confirming the refund status.
    """
    logging.info(f"Processing refund for order {order_id} of amount ${amount}")
    return json.dumps({
        "status": "success",
        "transaction_id": f"REF-{datetime.now().timestamp()}",
        "message": "Refund processed successfully.",
    })
