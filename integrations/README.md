# BlazeMetrics Integrations

This directory contains runnable integration examples for connecting BlazeMetrics to popular MLOps, monitoring, and LLM ecosystem tools.

## Integrations Included

- **wandb_integration.py**  
  Log BlazeMetrics evaluation metrics to Weights & Biases (wandb) for experiment tracking.

- **evidently_integration.py**  
  Use BlazeMetrics with Evidently for text drift and quality monitoring.

- **prometheus_integration.py**  
  Export BlazeMetrics metrics to Prometheus for production monitoring.

- **langchain_integration.py**  
  Use BlazeMetrics as a callback handler in a LangChain LLM pipeline.

- **llamaindex_integration.py**  
  Use BlazeMetrics as a callback handler in a LlamaIndex pipeline.

## How to Run

1. Install all dependencies:
   ```
   python3 -m pip install -r ../requirements.txt
   python3 -m pip install wandb evidently prometheus_client langchain llama-index
   ```

2. Run any script:
   ```
   PYTHONPATH=. python3 integrations/wandb_integration.py
   ```

## Notes

- Some scripts require external services or API keys (e.g., wandb, Prometheus, HuggingFace).
- Results will be printed to the console or logged to the respective service.
- For more focused or advanced examples, see the `/examples`, `/showcase`, and `/benchmarking` directories.

## Contributing

Feel free to add new integration scripts for other MLOps, monitoring, or LLM ecosystem tools.