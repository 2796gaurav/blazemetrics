# BlazeMetrics Examples

This directory contains runnable example scripts demonstrating BlazeMetrics' core features, advanced guardrails, distributed processing, async evaluation, and real-world use cases.

## Examples Included

- **metrics_basic.py**  
  Compute core NLP metrics (ROUGE, BLEU, METEOR, etc.) on sample data.

- **guardrails_basic.py**  
  Use blocklist, regex, and PII guardrails for content moderation.

- **streaming_analytics_basic.py**  
  Real-time streaming analytics and anomaly detection.

- **rag_semantic_basic.py**  
  RAG retrieval and semantic search with embeddings.

- **exporter_mlflow.py**  
  Log BlazeMetrics metrics to MLflow for experiment tracking.

- **integration_huggingface.py**  
  Evaluate HuggingFace pipeline outputs with BlazeMetrics.

- **custom_pii_patterns.py**  
  Add and use custom PII patterns for redaction and detection.

- **json_schema_validation.py**  
  Validate and repair structured LLM outputs using JSON schema guardrails.

- **fuzzy_unicode_blocklist.py**  
  Block obfuscated and Unicode-variant bad words.

- **multilingual_evaluation.py**  
  Metrics and guardrails on non-English texts.

- **async_batch_processing.py**  
  Async batch evaluation for large-scale processing.

- **ray_distributed_processing.py**  
  Distributed evaluation using Ray.

- **advanced_rag_evaluation.py**  
  Advanced RAG/semantic search and reranking.

## How to Run

1. Install all dependencies:
   ```
   python3 -m pip install -r ../requirements.txt
   ```

2. Run any script:
   ```
   PYTHONPATH=. python3 examples/metrics_basic.py
   ```

## Notes

- Some scripts require external services or API keys (e.g., HuggingFace, wandb).
- Results will be printed to the console.
- For more advanced or production use cases, see the `/showcase`, `/integrations`, and `/benchmarking` directories.

## Contributing

Feel free to add new examples or real-world scenarios to further demonstrate BlazeMetrics' capabilities.