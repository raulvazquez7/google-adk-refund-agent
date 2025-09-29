import json
import logging
import os
from datetime import datetime

from google.cloud import aiplatform, firestore
from vertexai.generative_models import GenerativeModel

# --- Configuration ---
# Cargar las variables de entorno para acceder a los recursos de GCP
# y al nombre del modelo de embeddings.
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
ME_ENDPOINT_NAME = os.getenv("ME_ENDPOINT_NAME")
ME_DEPLOYED_INDEX_ID = os.getenv("ME_DEPLOYED_INDEX_ID")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
FIRESTORE_DATABASE_ID = "orders"  # El ID de tu base de datos de Firestore

# Inicializamos el SDK de Vertex AI y el cliente de Firestore
aiplatform.init(project=PROJECT_ID, location=LOCATION)
db = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DATABASE_ID)


def rag_search_tool(query: str) -> str:
    """
    Performs a semantic search on the company's refund policy.

    This tool takes a user's question, converts it into a vector embedding,
    and queries the Vertex AI Matching Engine to find the most relevant
    sections of the policy document.

    Args:
        query: The user's question about the refund policy.

    Returns:
        A string containing the most relevant context from the policy,
        or a message if no relevant information is found.
    """
    logging.info(f"Performing RAG search with query: '{query}'")
    try:
        model = GenerativeModel(EMBEDDINGS_MODEL)
        response = model.embed_content([query], output_dimensionality=768)
        query_embedding = response[0].values

        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=ME_ENDPOINT_NAME
        )

        search_results = endpoint.find_neighbors(
            queries=[query_embedding],
            deployed_index_id=ME_DEPLOYED_INDEX_ID,
            num_neighbors=3,
        )

        context_pieces = []
        if search_results and search_results[0]:
            for match in search_results[0]:
                doc_text = next(
                    (r.string_value for r in match.datapoint.restricts if r.namespace == "text"),
                    None,
                )
                if doc_text:
                    context_pieces.append(doc_text)
        
        if not context_pieces:
            return "No se encontró información relevante en la política de devoluciones."

        logging.info(f"RAG search found {len(context_pieces)} relevant chunks.")
        return "\n---\n".join(context_pieces)

    except Exception as e:
        logging.error(f"Error during RAG search: {e}")
        return "Hubo un error al consultar la política de devoluciones."


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
