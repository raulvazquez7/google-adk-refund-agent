"""
Sets up Firestore Vector Search by generating and storing text embeddings.

This script reads the company's refund policy, splits it into chunks,
generates embeddings using Vertex AI, and stores them in Firestore
for semantic search capabilities.

Before running:
- Ensure GCP_PROJECT_ID, GCP_LOCATION, and EMBEDDINGS_MODEL are set in .env
- Authenticate with: gcloud auth application-default login
"""
import os
import json
from dotenv import load_dotenv
from google.cloud import firestore
from vertexai.language_models import TextEmbeddingModel
import numpy as np

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
FIRESTORE_DATABASE_ID = "orders"  # Use the same database for everything

def read_and_chunk_policy():
    """Reads and splits the policy document into chunks"""
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "company_refund_policy_barefoot.md")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by sections (###)
    chunks = content.split("### ")
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def generate_embeddings_and_store():
    """Generates embeddings and stores them in Firestore"""
    print("📚 Reading policy document...")
    chunks = read_and_chunk_policy()
    print(f"✅ Found {len(chunks)} chunks")
    
    print("🤖 Generating embeddings with Vertex AI...")
    model = TextEmbeddingModel.from_pretrained(EMBEDDINGS_MODEL)
    embeddings = model.get_embeddings(chunks)
    
    print("💾 Saving to Firestore...")
    db = firestore.Client(project=PROJECT_ID, database="orders")  # Use the existing database
    
    # Clear previous collection if it exists
    collection_ref = db.collection("policy_chunks")
    for doc in collection_ref.stream():
        doc.reference.delete()
    
    # Save new chunks with embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_ref = collection_ref.document(f"chunk_{i}")
        doc_ref.set({
            "text": chunk,
            "embedding": embedding.values,  # List of floats
            "chunk_id": i
        })
        print(f"  ✓ Chunk {i+1}/{len(chunks)}")
    
    print("\n🎉 Done! Vector search configured in Firestore")
    print(f"📊 {len(chunks)} chunks available for search")

if __name__ == "__main__":
    generate_embeddings_and_store()
