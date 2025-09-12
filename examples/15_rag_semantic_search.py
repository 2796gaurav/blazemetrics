"""
15_rag_semantic_search.py

BlazeMetrics Example – RAG (Retrieval-Augmented Generation) & Semantic Search
----------------------------------------------------------------------------
Demonstrates efficient document/embedding retrieval and reranking for RAG and semantic search/FAQ solutions:
  - Fast vector similarity search (finds top-k similar docs to query)
  - RAG reranking (score both raw similarity and after reranking/filtering)
  - Returns indices, scores (for further downstream use)

Intended for:
  - Users building RAG pipelines, doc Q&A, semantic FAQ bots
  - Any workflow using embeddings for search
"""
import numpy as np
from blazemetrics import BlazeMetricsClient

# Generate dummy embeddings for query and corpus (e.g., from a model)
np.random.seed(42)
query = np.random.randn(768).astype(np.float32)
corpus = np.random.randn(10, 768).astype(np.float32)

client = BlazeMetricsClient()

# Fast semantic search
results = client.semantic_search(query[None, :], corpus, top_k=3)[0]
print("Semantic search top-3:")
for idx, score in results:
    print(f"  Index: {idx}, Score: {score:.3f}")

# Fast RAG retrieval with reranking
rag_results = client.rag_search(query, corpus, top_k=3)
print("\nRAG search (top-3) with reranking:")
for idx, sim, reranked in rag_results:
    print(f"  Index: {idx}, Similarity: {sim:.3f}, Reranked: {reranked:.3f}")
