import json
import logging
import os
import random
import string

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud import storage

# --- Configuration ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- GCP Configuration from Environment Variables ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# --- Validation ---
REQUIRED_VARS = ["GCP_PROJECT_ID", "GCP_LOCATION", "GCS_BUCKET_NAME"]
missing_vars = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing_vars:
    logging.error(
        f"Missing required environment variables: {', '.join(missing_vars)}."
    )
    exit(1)

# --- Constants ---
# A unique ID for this run to avoid name clashes in GCP
UID = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

# The GCS folder containing our embeddings file(s)
EMBEDDINGS_GCS_URI = f"gs://{GCS_BUCKET_NAME}/rag_embeddings"

# Names for our Matching Engine resources
INDEX_DISPLAY_NAME = f"barefoot-policy-index-{UID}"
ENDPOINT_DISPLAY_NAME = f"barefoot-policy-endpoint-{UID}"
DEPLOYED_INDEX_ID = f"barefoot_policy_deployed_{UID}"


def get_embedding_dimensions(bucket_name: str, prefix: str) -> int:
    """
    Dynamically determines the dimensionality of embeddings by streaming
    the first line from the JSONL file in GCS.
    """
    logging.info("Determining embedding dimensions from GCS file...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob_list = list(bucket.list_blobs(prefix=prefix))
    jsonl_blob = next(
        (blob for blob in blob_list if blob.name.endswith(".jsonl")), None
    )

    if not jsonl_blob:
        raise FileNotFoundError(f"No .jsonl file found in gs://{bucket_name}/{prefix}")

    # Open the blob as a text stream and read only the first line
    # This is robust to line length and avoids downloading the whole file.
    with jsonl_blob.open("r", encoding="utf-8") as f:
        first_line = f.readline()

    if not first_line:
        raise ValueError("The embeddings file seems to be empty.")

    first_embedding = json.loads(first_line)
    dimensions = len(first_embedding["embedding"])
    logging.info(f"Embeddings have {dimensions} dimensions.")
    return dimensions


def main():
    """Main workflow to create and deploy a Matching Engine Index."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    try:
        dimensions = get_embedding_dimensions(
            bucket_name=GCS_BUCKET_NAME, prefix="rag_embeddings"
        )
    except Exception as e:
        logging.error(f"Could not determine embedding dimensions: {e}")
        return

    # 1. === CREATE THE INDEX (The "Central Warehouse") ===
    # This is the resource that stores and organizes our vectors for fast search.
    # We use a Tree-AH index, which is optimized for speed.
    logging.info(f"Creating Matching Engine Index: {INDEX_DISPLAY_NAME}...")
    my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=EMBEDDINGS_GCS_URI,
        dimensions=dimensions,
        approximate_neighbors_count=10,  # Lower for speed, higher for accuracy
        distance_measure_type="DOT_PRODUCT_DISTANCE",
    )
    logging.info(f"Index created. Resource Name: {my_index.resource_name}")
    logging.info("The index is now being built. This can take up to 60 minutes.")

    # 2. === CREATE THE ENDPOINT (The "Public-Facing Service Desk") ===
    # This creates a network endpoint (a set of VMs) that will host our index
    # and listen for incoming query requests.
    logging.info(f"Creating Index Endpoint: {ENDPOINT_DISPLAY_NAME}...")
    my_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,  # Accessible from the public internet
    )
    logging.info(f"Endpoint created. Resource Name: {my_endpoint.resource_name}")

    # 3. === DEPLOY THE INDEX TO THE ENDPOINT ("Staffing the Desk") ===
    # This crucial step links our "warehouse" to our "service desk".
    # The endpoint loads the index into memory to serve queries at low latency.
    logging.info(f"Deploying index {my_index.name} to endpoint {my_endpoint.name}...")
    my_endpoint.deploy_index(
        index=my_index, deployed_index_id=DEPLOYED_INDEX_ID
    )
    logging.info("Index deployed successfully.")

    # --- Final Output ---
    print("\n" + "=" * 50)
    print("  Matching Engine Setup Complete!")
    print("=" * 50)
    print("\nIMPORTANT: Save these resource names for your agent's tool:\n")
    print(f"  Index Resource Name: {my_index.resource_name}")
    print(f"  Endpoint Resource Name: {my_endpoint.resource_name}")
    print(f"  Deployed Index ID: {DEPLOYED_INDEX_ID}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
