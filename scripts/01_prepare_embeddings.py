import json
import logging
import os
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

# Load environment variables from .env file
load_dotenv()

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- GCP Configuration from Environment Variables ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "textembedding-gecko@001")

# --- Validation ---
REQUIRED_VARS = ["GCP_PROJECT_ID", "GCP_LOCATION", "GCS_BUCKET_NAME"]
missing_vars = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing_vars:
    logging.error(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please set them in your .env file or export them."
    )
    exit(1)

# --- Constants ---
GCS_BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
INPUT_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "company_refund_policy_barefoot.md"
)
OUTPUT_FOLDER_GCS = "rag_embeddings"
OUTPUT_FILE_NAME = "refund_policy_embeddings.json"
OUTPUT_FILE_PATH_GCS = f"{GCS_BUCKET_URI}/{OUTPUT_FOLDER_GCS}/{OUTPUT_FILE_NAME}"


def read_and_chunk_policy(file_path: str) -> List[str]:
    """Reads the policy document and splits it into semantic chunks."""
    logging.info(f"Reading and chunking file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = content.split("### ")
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    except FileNotFoundError:
        logging.error(f"Input file not found at: {file_path}")
        return []


def generate_embeddings(chunks: List[str]) -> List[Dict[str, Any]]:
    logging.info(f"Initializing embedding model '{EMBEDDINGS_MODEL}'...")
    model = TextEmbeddingModel.from_pretrained(EMBEDDINGS_MODEL)

    logging.info(f"Generating {len(chunks)} embeddings in a single batch...")
    embeddings = model.get_embeddings(chunks)

    embeddings_with_text = [
        {
            "id": str(uuid.uuid4()),
            "embedding": embedding.values,
            "text": chunk,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    logging.info("Embeddings generated successfully.")
    return embeddings_with_text


def save_to_gcs(bucket_name: str, gcs_path: str, data: List[Dict[str, Any]]):
    """Saves the embeddings to a JSONL file and uploads it to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{OUTPUT_FOLDER_GCS}/{OUTPUT_FILE_NAME}"
    blob = bucket.blob(blob_name)

    jsonl_data = "\n".join(json.dumps(item) for item in data)

    logging.info(f"Uploading file to {gcs_path}...")
    blob.upload_from_string(jsonl_data, content_type="application/jsonl")
    logging.info("File uploaded to GCS successfully.")


def main():
    """Main synchronous workflow."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    policy_chunks = read_and_chunk_policy(INPUT_FILE_PATH)
    if not policy_chunks:
        logging.error("No chunks were created from the policy file. Exiting.")
        return
    embeddings_data = generate_embeddings(policy_chunks)
    save_to_gcs(
        bucket_name=GCS_BUCKET_NAME,
        gcs_path=OUTPUT_FILE_PATH_GCS,
        data=embeddings_data,
    )
    logging.info("--- Process Completed ---")
    logging.info(f"Embeddings are available at: {OUTPUT_FILE_PATH_GCS}")


if __name__ == "__main__":
    main()
