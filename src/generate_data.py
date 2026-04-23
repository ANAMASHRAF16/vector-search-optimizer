"""
Generate sample documents for the vector search system.

Creates 500 documents with random embeddings (simulating real document vectors).
Saves them as .npy files that the search system loads.
"""

import os
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(BASE, exist_ok=True)

NUM_DOCS = 500
EMBEDDING_DIM = 128

# --- Generate random document embeddings ---
# In a real system these would come from a model like sentence-transformers
np.random.seed(42)
embeddings = np.random.rand(NUM_DOCS, EMBEDDING_DIM).astype("float32")

# Normalize vectors (required for cosine similarity search)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

# --- Generate document metadata ---
titles = [f"Document {i}" for i in range(NUM_DOCS)]

# Save
np.save(os.path.join(BASE, "embeddings.npy"), embeddings)
np.save(os.path.join(BASE, "titles.npy"), np.array(titles))

print(f"Generated {NUM_DOCS} documents with {EMBEDDING_DIM}-dim embeddings")
print(f"Saved to {BASE}/")
