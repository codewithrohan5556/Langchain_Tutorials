from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from transformers import logging

# Suppress warnings (optional)
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

# Initialize embeddings
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},  # or 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': False}  # Set to True for cosine similarity
)

documents = [
    "Mumbai is the capital of Maharashtra",
    "Delhi is the capital of India",
    "The financial capital of India is Mumbai"
]

# Generate embeddings for multiple documents
vectors = embedding.embed_documents(documents)

# Generate embedding for a single query
query = "What is the capital of Maharashtra?"
query_vector = embedding.embed_query(query)

# Print results
print(f"Number of documents: {len(vectors)}")
print(f"Embedding dimension: {len(vectors[0])}")
print(f"\nFirst document: {documents[0]}")
print(f"First 10 values of its embedding: {vectors[0][:10]}...")
print(f"\nQuery: {query}")
print(f"First 10 values of query embedding: {query_vector[:10]}...")

# Optional: Calculate similarity between query and documents
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n--- Similarity Scores ---")
for i, doc in enumerate(documents):
    similarity = cosine_similarity(query_vector, vectors[i])
    print(f"Document {i+1}: '{doc}'")
    print(f"Similarity to query: {similarity:.4f}\n")