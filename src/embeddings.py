# src/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a free local embedding model (no API key required)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    """
    Returns a 384-dimensional embedding vector using a free HuggingFace model.
    """
    if not isinstance(text, str):
        text = ""

    emb = model.encode([text], convert_to_numpy=True)[0]
    return np.array(emb)
