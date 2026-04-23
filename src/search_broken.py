"""
Vector Search -- BROKEN version.

PROBLEM: Rebuilds the FAISS index from scratch on EVERY search request.
This means every query pays the full cost of:
  1. Loading documents from disk
  2. Computing/loading embeddings
  3. Building the FAISS index
  4. Then finally searching

Result: >500ms per search, even for small datasets.
"""

import os
import time
import numpy as np
import faiss


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def search(query_text, top_k=5):
    """Search for similar documents. Rebuilds index every time (SLOW)."""
    start = time.time()

    # Step 1: Load embeddings from disk (every request!)
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    titles = np.load(os.path.join(DATA_DIR, "titles.npy"), allow_pickle=True)

    # Step 2: Build FAISS index from scratch (every request!)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim on normalized vecs)
    index.add(embeddings)

    # Step 3: Create a fake query embedding (simulate encoding the query)
    np.random.seed(hash(query_text) % 2**31)
    query_vec = np.random.rand(1, dim).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Step 4: Search
    scores, indices = index.search(query_vec, top_k)

    elapsed_ms = (time.time() - start) * 1000

    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        results.append({"rank": i + 1, "title": titles[idx], "score": float(score)})

    return results, elapsed_ms


# --- Run demo searches ---
if __name__ == "__main__":
    queries = ["machine learning basics", "data pipeline architecture", "cloud computing",
               "neural networks", "database optimization"]

    print("=== BROKEN VERSION (rebuilds index every request) ===\n")

    total_ms = 0
    for q in queries:
        results, ms = search(q)
        total_ms += ms
        print(f"Query: '{q}' -- {ms:.1f}ms")
        for r in results[:3]:
            print(f"  #{r['rank']} {r['title']} (score: {r['score']:.4f})")
        print()

    avg_ms = total_ms / len(queries)
    print(f"Average latency: {avg_ms:.1f}ms")
    if avg_ms > 100:
        print("SLOW: Latency exceeds 100ms target because index is rebuilt every request.")
