"""
Example: RAG and Semantic Search with BlazeMetrics

This script demonstrates how to use RAG retrieval and semantic search features.
"""

from blazemetrics.client import BlazeMetricsClient

client = BlazeMetricsClient()

import numpy as np

# Example: random embeddings for queries and corpus
query_emb = np.random.rand(1, 384).astype(np.float32)
corpus_emb = np.random.rand(3, 384).astype(np.float32)

# Semantic search
semantic_results = client.semantic_search(query_emb, corpus_emb, top_k=2)
print("Semantic Search Results:", semantic_results)

# RAG search (using the same dummy embeddings)
rag_results = client.rag_search(query_emb, corpus_emb, top_k=2, threshold=0.5)
print("RAG Search Results:", rag_results)