# BlazeMetrics Showcase

This directory contains comprehensive, end-to-end showcase scripts demonstrating BlazeMetrics' unique strengths in real-world, production-grade scenarios.

## Showcases Included

- **llm_e2e_showcase.py**  
  End-to-end LLM evaluation pipeline: generation, metrics, guardrails, streaming analytics, and MLflow export.

- **streaming_alerts_showcase.py**  
  Real-time streaming analytics and alerting for continuous evaluation.

- **adversarial_injection_showcase.py**  
  Detection of prompt injection, Unicode spoofing, and adversarial attacks in LLM pipelines.

- **streaming_moderation_showcase.py**  
  Real-time chat/content moderation with rolling window analytics and alerts.

## How to Run

1. Install all dependencies:
   ```
   python3 -m pip install -r ../requirements.txt
   ```

2. Run any script:
   ```
   PYTHONPATH=. python3 showcase/llm_e2e_showcase.py
   ```

## Notes

- Some scripts require external services or API keys (e.g., MLflow, HuggingFace).
- Results will be printed to the console.
- For more focused or integration-specific examples, see the `/examples`, `/integrations`, and `/benchmarking` directories.

## Contributing

Feel free to add new showcase scripts for advanced, real-world, or production scenarios.