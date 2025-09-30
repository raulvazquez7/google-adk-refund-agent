"""
Seeds the Firestore database with sample order data from a JSONL file.

This script reads order data from a local JSONL file and uploads each entry
as a document to a specified Firestore collection. The 'order_id' field is
used as the document ID in Firestore.

Before running:
- Ensure GCP_PROJECT_ID is set in your .env file
- Authenticate with: gcloud auth application-default login
"""
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from google.cloud import firestore

# --- Configuration ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- GCP Configuration ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
INPUT_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "orders.jsonl"
)
FIRESTORE_COLLECTION = "orders"

# --- Validation ---
if not PROJECT_ID:
    logging.error("GCP_PROJECT_ID not found in environment variables.")
    exit(1)


def seed_firestore():
    """
    Reads a JSONL file and uploads each line as a document to a Firestore collection.
    The 'order_id' from the JSON is used as the document ID in Firestore.
    """
    logging.info(f"Initializing Firestore client for project '{PROJECT_ID}'...")
    # Connect to the 'orders' database (not the default one)
    db = firestore.Client(project=PROJECT_ID, database="orders")
    collection_ref = db.collection(FIRESTORE_COLLECTION)

    logging.info(f"Reading data from '{INPUT_FILE_PATH}'...")
    try:
        with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    order_data = json.loads(line)
                    order_id = order_data.get("order_id")

                    if not order_id:
                        logging.warning(f"Skipping line due to missing 'order_id': {line.strip()}")
                        continue
                    
                    # Convert purchase_date string to a proper Firestore timestamp
                    iso_date_str = order_data.get("purchase_date")
                    if iso_date_str:
                        # Firestore client handles ISO 8601 strings automatically when writing
                        order_data["purchase_date"] = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))

                    doc_ref = collection_ref.document(order_id)
                    doc_ref.set(order_data)
                    logging.info(f"Successfully wrote document: {order_id}")

                except json.JSONDecodeError:
                    logging.error(f"Could not decode JSON from line: {line.strip()}")
                except Exception as e:
                    logging.error(f"An error occurred processing line for {order_id}: {e}")

        logging.info("--- Firestore seeding completed successfully! ---")

    except FileNotFoundError:
        logging.error(f"Input file not found at: {INPUT_FILE_PATH}")
        exit(1)


if __name__ == "__main__":
    seed_firestore()
