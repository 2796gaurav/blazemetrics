"""
Benchmarking: BlazeMetrics RAG/Semantic Search vs. sklearn NearestNeighbors

Compares BlazeMetrics' semantic search and RAG retrieval to sklearn's NearestNeighbors on large batches of embeddings.
"""

import time
import numpy as np
from blazemetrics.client import BlazeMetricsClient

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    print("scikit-learn not installed. Install with 'pip install scikit-learn'")
    NearestNeighbors = None

client = BlazeMetricsClient()

# Generate random embeddings for queries and corpus
num_queries = 100
corpus_size = 1000
embedding_dim = 384
corpus_emb = np.random.rand(corpus_size, embedding_dim).astype(np.float32)
query_emb = np.random.rand(num_queries, embedding_dim).astype(np.float32)

print("Benchmarking BlazeMetrics semantic_search...")
start = time.time()
bm_results = client.semantic_search(query_emb, corpus_emb, top_k=5)
bm_time = time.time() - start
print(f"BlazeMetrics semantic_search time: {bm_time:.3f}s")

if NearestNeighbors:
    print("Benchmarking sklearn NearestNeighbors...")
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    start = time.time()
    nn.fit(corpus_emb)
    distances, indices = nn.kneighbors(query_emb)
    nn_time = time.time() - start
    print(f"sklearn NearestNeighbors time: {nn_time:.3f}s")
else:
    indices = None

print("\n--- Results ---")
print("BlazeMetrics semantic_search (first query):", bm_results[0])
if indices is not None:
    print("sklearn NearestNeighbors (first query):", list(indices[0]))