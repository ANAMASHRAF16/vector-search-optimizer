"""
Vector Search -- FIXED version.

FIX: Load the FAISS index ONCE at startup, keep it in memory, and refresh
in the background. Searches hit the in-memory index directly.

Result: <100ms per search (typically <5ms).
"""

import os
import time
import threading
import numpy as np
import faiss


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class VectorSearchEngine:
    """In-memory vector search with background refresh."""

    def __init__(self, data_dir, refresh_interval=30):
        self.data_dir = data_dir
        self.refresh_interval = refresh_interval

        # Shared state
        self.index = None
        self.titles = None
        self.dim = None
        self._lock = threading.RLock()  # Protects index during refresh

        # Load index once at startup
        self._load_index()

        # Start background refresh thread
        self._stop_event = threading.Event()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _load_index(self):
        """Load embeddings and build FAISS index."""
        embeddings = np.load(os.path.join(self.data_dir, "embeddings.npy"))
        titles = np.load(os.path.join(self.data_dir, "titles.npy"), allow_pickle=True)

        dim = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(embeddings)

        # Swap in the new index atomically under the lock
        with self._lock:
            self.index = new_index
            self.titles = titles
            self.dim = dim

    def _refresh_loop(self):
        """Background thread: reload index every refresh_interval seconds."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.refresh_interval)
            if self._stop_event.is_set():
                break
            try:
                self._load_index()
            except Exception as e:
                print(f"[refresh] Error reloading index: {e}")

    def search(self, query_text, top_k=5):
        """Search the in-memory index (FAST -- no rebuild)."""
        start = time.time()

        # Create query vector
        np.random.seed(hash(query_text) % 2**31)
        query_vec = np.random.rand(1, self.dim).astype("float32")
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Search under read lock
        with self._lock:
            scores, indices = self.index.search(query_vec, top_k)
            titles = self.titles  # snapshot reference

        elapsed_ms = (time.time() - start) * 1000

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({"rank": i + 1, "title": titles[idx], "score": float(score)})

        return results, elapsed_ms

    def stop(self):
        """Stop the background refresh thread."""
        self._stop_event.set()
        self._refresh_thread.join(timeout=5)


# --- Run demo searches ---
if __name__ == "__main__":
    queries = ["machine learning basics", "data pipeline architecture", "cloud computing",
               "neural networks", "database optimization"]

    print("=== FIXED VERSION (in-memory index with background refresh) ===\n")

    # Index is built ONCE here
    engine = VectorSearchEngine(DATA_DIR, refresh_interval=30)
    print(f"Index loaded: {engine.index.ntotal} vectors, {engine.dim} dimensions\n")

    total_ms = 0
    for q in queries:
        results, ms = engine.search(q)
        total_ms += ms
        print(f"Query: '{q}' -- {ms:.1f}ms")
        for r in results[:3]:
            print(f"  #{r['rank']} {r['title']} (score: {r['score']:.4f})")
        print()

    avg_ms = total_ms / len(queries)
    print(f"Average latency: {avg_ms:.1f}ms")
    if avg_ms < 100:
        print("FAST: Latency under 100ms target. Index stays in memory.")

    engine.stop()
