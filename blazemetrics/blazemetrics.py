"""
BlazeMetrics Rust Extension Module

This module provides access to the compiled Rust extension functions.
"""

# Import the compiled Rust extension
try:
    import blazemetrics.blazemetrics_abi3 as _ext
except ImportError:
    try:
        import blazemetrics.blazemetrics as _ext
    except ImportError:
        raise ImportError(
            "BlazeMetrics Rust extension not found. "
            "Please ensure the extension is properly compiled and installed."
        )

# Re-export all functions from the extension
__all__ = [
    "rouge_score",
    "bleu",
    "chrf_score",
    "token_f1",
    "jaccard",
    "moverscore_greedy_py",
    "meteor",
    "wer",
    "guard_blocklist",
    "guard_regex",
    "guard_pii_redact",
    "guard_safety_score",
    "guard_json_validate",
    "guard_detect_injection_spoof",
    "guard_max_cosine_similarity",
    "guard_fuzzy_blocklist",
    "guard_fuzzy_blocklist_detailed",
    "batch_cosine_similarity_optimized",
    "semantic_search_topk",
    "rag_retrieval_with_reranking",
]

# Import all functions
(
    rouge_score,
    bleu,
    chrf_score,
    token_f1,
    jaccard,
    moverscore_greedy_py,
    meteor,
    wer,
    guard_blocklist,
    guard_regex,
    guard_pii_redact,
    guard_safety_score,
    guard_json_validate,
    guard_detect_injection_spoof,
    guard_max_cosine_similarity,
    guard_fuzzy_blocklist,
    guard_fuzzy_blocklist_detailed,
    batch_cosine_similarity_optimized,
    semantic_search_topk,
    rag_retrieval_with_reranking,
) = (
    _ext.rouge_score,
    _ext.bleu,
    _ext.chrf_score,
    _ext.token_f1,
    _ext.jaccard,
    _ext.moverscore_greedy_py,
    _ext.meteor,
    _ext.wer,
    _ext.guard_blocklist,
    _ext.guard_regex,
    _ext.guard_pii_redact,
    _ext.guard_safety_score,
    _ext.guard_json_validate,
    _ext.guard_detect_injection_spoof,
    _ext.guard_max_cosine_similarity,
    _ext.guard_fuzzy_blocklist,
    _ext.guard_fuzzy_blocklist_detailed,
    _ext.batch_cosine_similarity_optimized,
    _ext.semantic_search_topk,
    _ext.rag_retrieval_with_reranking,
) 