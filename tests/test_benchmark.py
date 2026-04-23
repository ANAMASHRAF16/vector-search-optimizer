"""
Benchmark: Compare broken vs fixed vector search.

Runs both versions and asserts the fixed version meets the <100ms target.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search_broken import search as broken_search
from search_fixed import VectorSearchEngine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

QUERIES = [
    "machine learning basics",
    "data pipeline architecture",
    "cloud computing setup",
    "neural network training",
    "database optimization tips",
    "kubernetes deployment",
    "api design patterns",
    "security best practices",
    "monitoring and alerting",
    "ci cd pipeline setup",
]

NUM_RUNS = 3  # Run each query multiple times for stable measurement


def benchmark_broken():
    """Benchmark the broken version (rebuilds index every request)."""
    latencies = []
    for _ in range(NUM_RUNS):
        for q in QUERIES:
            _, ms = broken_search(q)
            latencies.append(ms)
    return latencies


def benchmark_fixed():
    """Benchmark the fixed version (in-memory index)."""
    engine = VectorSearchEngine(DATA_DIR, refresh_interval=60)
    latencies = []
    for _ in range(NUM_RUNS):
        for q in QUERIES:
            _, ms = engine.search(q)
            latencies.append(ms)
    engine.stop()
    return latencies


if __name__ == "__main__":
    # Check data exists
    if not os.path.exists(os.path.join(DATA_DIR, "embeddings.npy")):
        print("No data found. Run 'python src/generate_data.py' first.")
        sys.exit(1)

    print("=" * 60)
    print("BENCHMARK: Broken vs Fixed Vector Search")
    print("=" * 60)

    # --- Broken ---
    print("\n[1/2] Running BROKEN version (rebuilds index every request)...")
    broken_latencies = benchmark_broken()
    broken_avg = sum(broken_latencies) / len(broken_latencies)
    broken_p99 = sorted(broken_latencies)[int(len(broken_latencies) * 0.99)]

    print(f"  Average: {broken_avg:.1f}ms")
    print(f"  P99:     {broken_p99:.1f}ms")
    print(f"  Min:     {min(broken_latencies):.1f}ms")
    print(f"  Max:     {max(broken_latencies):.1f}ms")

    # --- Fixed ---
    print("\n[2/2] Running FIXED version (in-memory index)...")
    fixed_latencies = benchmark_fixed()
    fixed_avg = sum(fixed_latencies) / len(fixed_latencies)
    fixed_p99 = sorted(fixed_latencies)[int(len(fixed_latencies) * 0.99)]

    print(f"  Average: {fixed_avg:.1f}ms")
    print(f"  P99:     {fixed_p99:.1f}ms")
    print(f"  Min:     {min(fixed_latencies):.1f}ms")
    print(f"  Max:     {max(fixed_latencies):.1f}ms")

    # --- Comparison ---
    speedup = broken_avg / fixed_avg if fixed_avg > 0 else float("inf")
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Broken avg:  {broken_avg:.1f}ms")
    print(f"  Fixed avg:   {fixed_avg:.1f}ms")
    print(f"  Speedup:     {speedup:.0f}x faster")
    print(f"{'=' * 60}")

    # --- Assertions ---
    passed = 0
    failed = 0

    # Test 1: Fixed version should be under 100ms average
    if fixed_avg < 100:
        print("PASS: Fixed version average latency < 100ms")
        passed += 1
    else:
        print(f"FAIL: Fixed version average latency {fixed_avg:.1f}ms >= 100ms")
        failed += 1

    # Test 2: Fixed should be at least 5x faster than broken
    if speedup >= 5:
        print(f"PASS: Fixed is {speedup:.0f}x faster (target: >= 5x)")
        passed += 1
    else:
        print(f"FAIL: Speedup only {speedup:.1f}x (target: >= 5x)")
        failed += 1

    # Test 3: Same results from both versions
    broken_results, _ = broken_search("test query")
    engine = VectorSearchEngine(DATA_DIR)
    fixed_results, _ = engine.search("test query")
    engine.stop()

    broken_titles = [r["title"] for r in broken_results]
    fixed_titles = [r["title"] for r in fixed_results]
    if broken_titles == fixed_titles:
        print("PASS: Both versions return identical results")
        passed += 1
    else:
        print("FAIL: Results differ between versions")
        failed += 1

    # Test 4: Background refresh doesn't crash
    try:
        engine = VectorSearchEngine(DATA_DIR, refresh_interval=1)
        time.sleep(2)  # Let refresh run at least once
        _, ms = engine.search("refresh test")
        engine.stop()
        print("PASS: Background refresh works without errors")
        passed += 1
    except Exception as e:
        print(f"FAIL: Background refresh error: {e}")
        failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
