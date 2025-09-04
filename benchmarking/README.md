# BlazeMetrics Benchmarking Suite

This directory contains benchmarking scripts to evaluate BlazeMetrics' performance, correctness, and robustness against popular open-source libraries and real-world scenarios.

## Benchmarks Included

- **text_metrics_benchmark.py**  
  Compare BlazeMetrics' ROUGE, BLEU, and WER to sacrebleu, rouge-score, and jiwer on large batches.

- **rag_semantic_benchmark.py**  
  Compare BlazeMetrics' semantic search and RAG retrieval to sklearn's NearestNeighbors.

- **huggingface_usecase_benchmark.py**  
  Evaluate BlazeMetrics metrics and guardrails on real HuggingFace text-generation outputs.

- **json_guardrails_benchmark.py**  
  Benchmark BlazeMetrics' speed and correctness for JSON schema validation and injection detection.

## How to Run

1. Install all dependencies:
   ```
   python3 -m pip install -r ../requirements.txt
   python3 -m pip install sacrebleu rouge-score jiwer scikit-learn transformers torch
   ```

2. Run each script:
   ```
   PYTHONPATH=. python3 benchmarking/text_metrics_benchmark.py
   PYTHONPATH=. python3 benchmarking/rag_semantic_benchmark.py
   PYTHONPATH=. python3 benchmarking/huggingface_usecase_benchmark.py
   PYTHONPATH=. python3 benchmarking/json_guardrails_benchmark.py
   ```

## Notes

- Some scripts require external services or API keys (e.g., HuggingFace, wandb).
- Results will be printed to the console. For large-scale or automated benchmarking, consider redirecting output to a file.
- If BlazeMetrics underperforms in any area, see the codebase for optimization opportunities or open an issue.

## Contributing

Feel free to add new benchmarks, real-world scenarios, or competitor libraries to further validate and improve BlazeMetrics.