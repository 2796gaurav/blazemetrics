"""
BlazeMetrics Client - Unified, Comprehensive API

This module provides a single, unified interface for all BlazeMetrics capabilities:
- Text Metrics (ROUGE, BLEU, chrF, METEOR, WER, etc.)
- Guardrails (Blocklist, Regex, PII, Safety, JSON validation)
- Fuzzy Matching (Levenshtein, Damerau-Levenshtein, Jaro-Winkler)
- Embedding Operations (RAG, Semantic Search, Similarity)
- Streaming Analytics (Real-time monitoring, alerts, trends)
- LLM Integrations (Provider-specific configurations)
- Monitoring & Exporting (Prometheus, StatsD, custom exporters)

All features are available via BlazeMetricsClient and ClientConfig.
"""

from typing import List, Dict, Any, Optional, Union, Literal, Callable, Tuple
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys

# Import core functionality
try:
    from . import (
        # Text metrics
        rouge_score, bleu, chrf_score, token_f1, jaccard, meteor, wer,
        # Fuzzy matching
        guard_fuzzy_blocklist, guard_fuzzy_blocklist_detailed,
        # Embedding operations
        batch_cosine_similarity_optimized, semantic_search_topk, rag_retrieval_with_reranking,
    )
    # Try to import guardrails functions
    try:
        from . import (
            guard_blocklist, guard_regex, guard_pii_redact, guard_safety_score,
            guard_json_validate, guard_detect_injection_spoof, guard_max_cosine_similarity,
        )
        GUARDRAILS_AVAILABLE = True
    except ImportError:
        GUARDRAILS_AVAILABLE = False
    # Try to import BERTScore functions
    try:
        from . import bert_score_similarity, moverscore_greedy
        BERTSCORE_AVAILABLE = True
    except ImportError:
        BERTSCORE_AVAILABLE = False
    HAS_RUST = True
except ImportError as e:
    HAS_RUST = False
    GUARDRAILS_AVAILABLE = False
    BERTSCORE_AVAILABLE = False
    warnings.warn(f"Rust extension not available: {e}. Using Python fallbacks.")

from .enhanced_guardrails import EnhancedGuardrails
from .streaming_analytics import StreamingAnalytics, create_llm_monitoring_analytics
from .llm_integrations import EnhancedPIIDetector, create_llm_guardrails
from .metrics import compute_text_metrics, aggregate_samples
from .monitor import monitor_stream_sync, monitor_stream_async
from .guardrails_pipeline import monitor_tokens_sync, monitor_tokens_async, map_large_texts, enforce_stream_sync
from .exporters import MetricsExporters

@dataclass
class ClientConfig:
    """Comprehensive configuration for the BlazeMetrics client"""
    # Text metrics
    metrics_include: List[str] = field(default_factory=lambda: [
        "rouge1", "rouge2", "rougeL", "bleu", "chrf", "meteor", "wer", "token_f1", "jaccard"
    ])
    metrics_lowercase: bool = False
    metrics_stemming: bool = False

    # Guardrails
    blocklist: List[str] = field(default_factory=list)
    regexes: List[str] = field(default_factory=list)
    case_insensitive: bool = True
    redact_pii: bool = True
    safety: bool = True
    json_schema: Optional[str] = None
    detect_injection: bool = True

    # Fuzzy matching
    fuzzy_distance: int = 2
    fuzzy_algorithm: Literal["levenshtein", "damerau_levenshtein", "jaro_winkler"] = "levenshtein"

    # PII detection
    detect_pii: bool = True
    enhanced_pii: bool = True

    # Analytics
    enable_analytics: bool = False
    analytics_window: int = 100
    analytics_alerts: bool = True
    analytics_trends: bool = True
    analytics_anomalies: bool = True

    # Monitoring
    enable_monitoring: bool = False
    monitoring_window: int = 100
    monitoring_thresholds: Dict[str, float] = field(default_factory=dict)

    # Exporting
    prometheus_gateway: Optional[str] = None
    statsd_addr: Optional[str] = None

    # LLM provider
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None

    # Performance
    parallel_processing: bool = True
    chunk_size: int = 1000
    max_workers: Optional[int] = None

class BlazeMetricsClient:
    """
    Unified client for ALL BlazeMetrics features.

    Usage Examples:

    # Basic setup
    client = BlazeMetricsClient()
    scores = client.compute_metrics(candidates, references)
    safety_results = client.check_safety(texts)
    results = client.rag_search(query, corpus)

    # Monitoring with analytics
    client = BlazeMetricsClient(enable_analytics=True, enable_monitoring=True)
    client.monitor_stream(prompts, references)
    """

    def __init__(self, config: Optional[ClientConfig] = None, **kwargs):
        """
        Initialize the client with comprehensive configuration.

        Args:
            config: ClientConfig object or use kwargs for quick setup
            **kwargs: Quick configuration options
        """
        if config is None:
            config = ClientConfig(**kwargs)

        self.config = config
        self._components = {}
        self._exporters = None

        # Initialize all components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all internal components based on configuration"""
        # Initialize guardrails
        if self.config.blocklist or self.config.regexes or self.config.detect_injection:
            try:
                self._components['guardrails'] = EnhancedGuardrails(
                    blocklist=self.config.blocklist,
                    fuzzy_blocklist=self.config.blocklist,
                    fuzzy_config={
                        "max_distance": self.config.fuzzy_distance,
                        "algorithm": self.config.fuzzy_algorithm,
                        "case_sensitive": not self.config.case_insensitive
                    },
                    regexes=self.config.regexes,
                    case_insensitive=self.config.case_insensitive,
                    redact_pii=self.config.redact_pii,
                    enhanced_pii=self.config.enhanced_pii,
                    safety=self.config.safety,
                    json_schema=self.config.json_schema,
                    detect_injection_spoof=self.config.detect_injection,
                    llm_provider=self.config.llm_provider,
                    model_name=self.config.model_name,
                    streaming_analytics=self.config.enable_analytics,
                    analytics_window_size=self.config.analytics_window
                )
            except NotImplementedError:
                from .guardrails import Guardrails
                self._components['guardrails'] = Guardrails(
                    blocklist=self.config.blocklist,
                    regexes=self.config.regexes,
                    case_insensitive=self.config.case_insensitive,
                    redact_pii=self.config.redact_pii,
                    safety=self.config.safety,
                    json_schema=self.config.json_schema,
                    detect_injection_spoof=self.config.detect_injection
                )

        # Initialize analytics
        if self.config.enable_analytics:
            provider = self.config.llm_provider or "generic"
            self._components['analytics'] = create_llm_monitoring_analytics(
                window_size=self.config.analytics_window,
                provider=provider
            )
            # Set up analytics callbacks
            if self.config.analytics_alerts:
                self._components['analytics'].on_alert = self._default_alert_callback
            if self.config.analytics_anomalies:
                self._components['analytics'].on_anomaly = self._default_anomaly_callback
            if self.config.analytics_trends:
                self._components['analytics'].on_trend = self._default_trend_callback

        # Initialize PII detector
        if self.config.detect_pii:
            self._components['pii_detector'] = EnhancedPIIDetector()

        # Initialize exporters
        if self.config.prometheus_gateway or self.config.statsd_addr:
            self._exporters = MetricsExporters(
                prometheus_gateway=self.config.prometheus_gateway,
                statsd_addr=self.config.statsd_addr
            )

    # ===== TEXT METRICS METHODS =====

    def compute_metrics(self, 
                       candidates: List[str], 
                       references: List[List[str]],
                       include: Optional[List[str]] = None,
                       lowercase: Optional[bool] = None,
                       stemming: Optional[bool] = None) -> Dict[str, List[float]]:
        """
        Compute comprehensive text evaluation metrics.
        """
        if not HAS_RUST:
            warnings.warn("Using Python fallback for text metrics")
            return compute_text_metrics(
                candidates, references, 
                include=include or self.config.metrics_include,
                lowercase=lowercase if lowercase is not None else self.config.metrics_lowercase,
                stemming=stemming if stemming is not None else self.config.metrics_stemming
            )

        # Use Rust implementations for individual metrics
        include = include or self.config.metrics_include
        lowercase = lowercase if lowercase is not None else self.config.metrics_lowercase
        stemming = stemming if stemming is not None else self.config.metrics_stemming

        # Normalize texts if needed
        if lowercase or stemming:
            candidates = self._normalize_texts(candidates, lowercase, stemming)
            references = [self._normalize_texts(refs, lowercase, stemming) for refs in references]

        results = {}

        # ROUGE metrics
        if any(m in include for m in ["rouge1", "rouge2", "rougeL"]):
            if "rouge1" in include:
                results["rouge1_f1"] = [t[2] for t in rouge_score(candidates, references, "rouge_n", 1)]
            if "rouge2" in include:
                results["rouge2_f1"] = [t[2] for t in rouge_score(candidates, references, "rouge_n", 2)]
            if "rougeL" in include:
                results["rougeL_f1"] = [t[2] for t in rouge_score(candidates, references, "rouge_l")]

        # Other metrics
        if "bleu" in include:
            results["bleu"] = bleu(candidates, references)
        if "chrf" in include:
            results["chrf"] = chrf_score(candidates, references)
        if "meteor" in include:
            results["meteor"] = meteor(candidates, references)
        if "wer" in include:
            results["wer"] = wer(candidates, references)
        if "token_f1" in include:
            results["token_f1"] = token_f1(candidates, references)
        if "jaccard" in include:
            results["jaccard"] = jaccard(candidates, references)

        return results

    def bert_score_similarity(self, 
                            candidates: List[str], 
                            references: List[List[str]],
                            model_type: str = "bert-base-uncased") -> List[float]:
        """Compute BERTScore similarity between candidates and references."""
        if not BERTSCORE_AVAILABLE:
            warnings.warn("BERTScore not available")
            return [0.0] * len(candidates)
        return bert_score_similarity(candidates, references, model_type)

    def moverscore(self, 
                   candidates: List[str], 
                   references: List[List[str]]) -> List[float]:
        """Compute MoverScore between candidates and references."""
        if not BERTSCORE_AVAILABLE:
            warnings.warn("MoverScore not available")
            return [0.0] * len(candidates)
        return moverscore_greedy(candidates, references)

    def aggregate_metrics(self, 
                         metrics: Dict[str, List[float]], 
                         weights: Optional[List[float]] = None) -> Dict[str, float]:
        """Aggregate metric scores across samples."""
        return aggregate_samples(metrics, weights)

    # ===== GUARDRAILS METHODS =====

    def check_safety(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Comprehensive safety check using all available guardrails.
        """
        if isinstance(texts, str):
            return self._check_single_safety(texts)
        else:
            return [self._check_single_safety(text) for text in texts]

    def fuzzy_blocklist(self, texts: List[str], patterns: List[str]) -> List[bool]:
        """Check texts against fuzzy blocklist patterns."""
        if not HAS_RUST:
            warnings.warn("Using Python fallback for fuzzy matching")
            return self._fallback_fuzzy_blocklist(texts, patterns)
        return guard_fuzzy_blocklist(
            texts, patterns,
            max_distance=self.config.fuzzy_distance,
            algorithm=self.config.fuzzy_algorithm,
            case_sensitive=not self.config.case_insensitive
        )

    def detect_pii(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Detect PII in texts."""
        if not self._components.get('pii_detector'):
            self._components['pii_detector'] = EnhancedPIIDetector()
        if isinstance(texts, str):
            return self._components['pii_detector'].detect_pii_batch([texts])[0]
        else:
            return self._components['pii_detector'].detect_pii_batch(texts)

    def validate_json(self, texts: List[str], schema: str) -> Tuple[List[bool], List[str]]:
        """Validate JSON texts against schema."""
        if not HAS_RUST:
            warnings.warn("JSON validation not available without Rust extension")
            return [False] * len(texts), [""] * len(texts)
        return guard_json_validate(texts, schema)

    def detect_injection(self, texts: List[str]) -> List[bool]:
        """Detect injection/spoofing attempts."""
        if not HAS_RUST:
            warnings.warn("Injection detection not available without Rust extension")
            return [False] * len(texts)
        return guard_detect_injection_spoof(texts)

    # ===== EMBEDDING & RAG METHODS =====

    def rag_search(self, 
                  query: npt.NDArray[np.float32], 
                  corpus: npt.NDArray[np.float32],
                  top_k: int = 5,
                  threshold: float = 0.1) -> List[tuple]:
        """Fast RAG search with reranking."""
        if not HAS_RUST:
            warnings.warn("Using Python fallback for RAG search")
            return self._fallback_rag_search(query, corpus, top_k, threshold)
        return rag_retrieval_with_reranking(query, corpus, top_k, threshold)

    def semantic_search(self, 
                       queries: npt.NDArray[np.float32],
                       corpus: npt.NDArray[np.float32],
                       top_k: int = 5) -> List[List[tuple]]:
        """Semantic search with top-k results."""
        if not HAS_RUST:
            warnings.warn("Using Python fallback for semantic search")
            return self._fallback_semantic_search(queries, corpus, top_k)
        return semantic_search_topk(queries, corpus, top_k)

    def batch_similarity(self, 
                        embeddings1: npt.NDArray[np.float32],
                        embeddings2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Batch cosine similarity between embeddings."""
        if not HAS_RUST:
            warnings.warn("Using Python fallback for batch similarity")
            return self._fallback_batch_similarity(embeddings1, embeddings2)
        result = batch_cosine_similarity_optimized(embeddings1, embeddings2)
        if isinstance(result, list):
            return np.array(result)
        return result

    def max_similarity_to_unsafe(self, 
                                candidates: List[List[float]], 
                                exemplars: List[List[float]]) -> List[float]:
        """Find maximum cosine similarity to unsafe exemplars."""
        if not HAS_RUST:
            warnings.warn("Using Python fallback for max similarity")
            return self._fallback_max_similarity(candidates, exemplars)
        c = np.array(candidates, dtype=np.float32)
        e = np.array(exemplars, dtype=np.float32)
        return guard_max_cosine_similarity(c, e)

    # ===== MONITORING & ANALYTICS METHODS =====

    def monitor_stream(self, 
                      stream: List[Tuple[str, List[str]]],
                      window_size: Optional[int] = None,
                      include: Optional[List[str]] = None,
                      thresholds: Optional[Dict[str, float]] = None,
                      lowercase: Optional[bool] = None,
                      stemming: Optional[bool] = None) -> None:
        """Monitor streaming text generation with metrics."""
        window_size = window_size or self.config.monitoring_window
        include = include or self.config.metrics_include
        thresholds = thresholds or self.config.monitoring_thresholds
        lowercase = lowercase if lowercase is not None else self.config.metrics_lowercase
        stemming = stemming if stemming is not None else self.config.metrics_stemming

        if self.config.enable_monitoring:
            monitor_stream_sync(
                stream, window_size, include, thresholds, lowercase, stemming,
                prometheus_gateway=self.config.prometheus_gateway,
                statsd_addr=self.config.statsd_addr
            )
        else:
            # Just compute metrics without monitoring
            candidates = [prompt for prompt, _ in stream]
            references = [refs for _, refs in stream]
            self.compute_metrics(candidates, references, include, lowercase, stemming)

    async def monitor_stream_async(self, 
                                 stream: List[Tuple[str, List[str]]],
                                 window_size: Optional[int] = None,
                                 include: Optional[List[str]] = None,
                                 thresholds: Optional[Dict[str, float]] = None,
                                 lowercase: Optional[bool] = None,
                                 stemming: Optional[bool] = None,
                                 delay_s: float = 0.0) -> None:
        """Asynchronously monitor streaming text generation."""
        window_size = window_size or self.config.monitoring_window
        include = include or self.config.metrics_include
        thresholds = thresholds or self.config.monitoring_thresholds
        lowercase = lowercase if lowercase is not None else self.config.metrics_lowercase
        stemming = stemming if stemming is not None else self.config.metrics_stemming

        if self.config.enable_monitoring:
            await monitor_stream_async(
                stream, window_size, include, thresholds, lowercase, stemming,
                prometheus_gateway=self.config.prometheus_gateway,
                statsd_addr=self.config.statsd_addr,
                delay_s=delay_s
            )
        else:
            # Just compute metrics without monitoring
            candidates = [prompt for prompt, _ in stream]
            references = [refs for _, refs in stream]
            self.compute_metrics(candidates, references, include, lowercase, stemming)

    def monitor_tokens(self, 
                      tokens: List[str],
                      every_n_tokens: int = 20,
                      joiner: str = "") -> List[Dict[str, Any]]:
        """Monitor token-level generation with guardrails."""
        if not self._components.get('guardrails'):
            return []
        return list(monitor_tokens_sync(
            tokens, self._components['guardrails'], every_n_tokens, joiner
        ))

    def process_large_texts(self, 
                           texts: List[str],
                           chunk_size: Optional[int] = None,
                           max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process large texts in parallel chunks."""
        if not self._components.get('guardrails'):
            return []
        chunk_size = chunk_size or self.config.chunk_size
        max_workers = max_workers or self.config.max_workers
        return map_large_texts(
            texts, self._components['guardrails'], max_workers, chunk_size
        )

    def enforce_safety_stream(self, 
                             tokens: List[str],
                             every_n_tokens: int = 20,
                             joiner: str = "",
                             replacement: str = "[REDACTED]",
                             safety_threshold: float = 0.7,
                             on_violation: Optional[Callable] = None) -> List[str]:
        """Enforce safety on token stream with automatic redaction."""
        if not self._components.get('guardrails'):
            return tokens
        return list(enforce_stream_sync(
            tokens, self._components['guardrails'], every_n_tokens, joiner,
            replacement, safety_threshold, on_violation
        ))

    # ===== ANALYTICS METHODS =====

    def add_metrics(self, metrics: Dict[str, float], metadata: Optional[Dict] = None):
        """Add metrics to analytics if enabled."""
        if self._components.get('analytics'):
            # Only include numeric metadata
            clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        clean_metrics[key] = value
            self._components['analytics'].add_metrics(clean_metrics)

    def get_analytics_summary(self) -> Optional[Dict[str, Any]]:
        """Get analytics summary if enabled."""
        if self._components.get('analytics'):
            return self._components['analytics'].get_metric_summary()
        return None

    def export_metrics(self, metrics: Dict[str, float], labels: Optional[Dict[str, str]] = None):
        """Export metrics to configured destinations."""
        if self._exporters:
            self._exporters.export(metrics, labels)

    # ===== UTILITY METHODS =====

    def _normalize_texts(self, texts: List[str], lowercase: bool, stemming: bool) -> List[str]:
        """Normalize texts for metric computation."""
        if not lowercase and not stemming:
            return texts
        try:
            from nltk.stem.porter import PorterStemmer
            stemmer = PorterStemmer() if stemming else None
        except ImportError:
            stemmer = None
            if stemming:
                warnings.warn("NLTK not available, skipping stemming")
        normalized = []
        for text in texts:
            normalized_text = text.lower() if lowercase else text
            if stemmer:
                normalized_text = " ".join(stemmer.stem(tok) for tok in normalized_text.split())
            normalized.append(normalized_text)
        return normalized

    def _check_single_safety(self, text: str) -> Dict[str, Any]:
        """Check safety for a single text."""
        if not self._components.get('guardrails'):
            return {"blocked": False, "safe": True, "guardrails_available": False}
        try:
            result = self._components['guardrails'].check([text])
            return result[0] if isinstance(result, list) else result
        except Exception as e:
            warnings.warn(f"Safety check failed: {e}")
            return {"blocked": False, "safe": True, "error": str(e), "guardrails_available": False}

    # ===== FALLBACK IMPLEMENTATIONS =====

    def _fallback_fuzzy_blocklist(self, texts: List[str], patterns: List[str]) -> List[bool]:
        """Fallback fuzzy blocklist implementation."""
        results = []
        for text in texts:
            text_lower = text.lower()
            blocked = False
            for pattern in patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in text_lower:
                    blocked = True
                    break
            results.append(blocked)
        return results

    def _fallback_rag_search(self, query, corpus, top_k, threshold):
        """Fallback RAG search implementation."""
        similarities = self._fallback_batch_similarity(query, corpus)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        results = []
        for idx in top_indices:
            sim = similarities[0][idx]
            if sim >= threshold:
                results.append((int(idx), float(sim), float(sim)))
        return results

    def _fallback_semantic_search(self, queries, corpus, top_k):
        """Fallback semantic search implementation."""
        results = []
        for query in queries:
            similarities = self._fallback_batch_similarity(query.reshape(1, -1), corpus)
            top_indices = np.argsort(similarities[0])[-top_k:][::-1]
            query_results = []
            for idx in top_indices:
                sim = similarities[0][idx]
                query_results.append((int(idx), float(sim)))
            results.append(query_results)
        return results

    def _fallback_batch_similarity(self, emb1, emb2):
        """Fallback batch similarity implementation."""
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        return np.dot(emb1_norm, emb2_norm.T)

    def _fallback_max_similarity(self, candidates, exemplars):
        """Fallback max similarity implementation."""
        candidates_array = np.array(candidates, dtype=np.float32)
        exemplars_array = np.array(exemplars, dtype=np.float32)
        # Normalize
        candidates_norm = candidates_array / np.linalg.norm(candidates_array, axis=1, keepdims=True)
        exemplars_norm = exemplars_array / np.linalg.norm(exemplars_array, axis=1, keepdims=True)
        # Compute similarities
        similarities = np.dot(candidates_norm, exemplars_norm.T)
        return np.max(similarities, axis=1).tolist()

    # ===== CALLBACK METHODS =====

    def _default_alert_callback(self, alert):
        """Default alert callback for analytics."""
        print(f"🚨 ALERT [{alert.severity.upper()}]: {alert.message}")

    def _default_anomaly_callback(self, metric_name, value, anomalies):
        """Default anomaly callback for analytics."""
        print(f"⚠️  ANOMALY: {metric_name} = {value:.3f}")

    def _default_trend_callback(self, metric_name, trend):
        """Default trend callback for analytics."""
        if abs(trend) > 0.1:
            direction = "↗️" if trend > 0 else "↘️"
            print(f"📈 TREND: {metric_name} {direction} (slope: {trend:.3f})")

    # ===== COMPONENT ACCESS =====

    def get_guardrails(self) -> Optional[EnhancedGuardrails]:
        """Get the underlying guardrails instance."""
        return self._components.get('guardrails')

    def get_analytics(self) -> Optional[StreamingAnalytics]:
        """Get the underlying analytics instance."""
        return self._components.get('analytics')

    def get_pii_detector(self) -> Optional[EnhancedPIIDetector]:
        """Get the underlying PII detector instance."""
        return self._components.get('pii_detector')

    def get_exporters(self) -> Optional[MetricsExporters]:
        """Get the underlying exporters instance."""
        return self._exporters

# ===== CONVENIENCE FUNCTIONS =====

def quick_metrics(candidates: List[str], 
                 references: List[List[str]], 
                 include: Optional[List[str]] = None) -> Dict[str, List[float]]:
    """Quick text metrics computation without creating client."""
    client = BlazeMetricsClient()
    return client.compute_metrics(candidates, references, include)

def quick_safety_check(texts: List[str], 
                      blocklist: Optional[List[str]] = None,
                      regexes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Quick safety check without creating client."""
    config = ClientConfig()
    if blocklist:
        config.blocklist = blocklist
    if regexes:
        config.regexes = regexes
    client = BlazeMetricsClient(config)
    return client.check_safety(texts)

def quick_rag_search(query: npt.NDArray[np.float32],
                    corpus: npt.NDArray[np.float32],
                    top_k: int = 5) -> List[tuple]:
    """Quick RAG search without creating client."""
    client = BlazeMetricsClient()
    return client.rag_search(query, corpus, top_k)