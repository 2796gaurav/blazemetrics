"""
12_hf_eval_and_openai_example.py

BlazeMetrics Example – HuggingFace & OpenAI LLM Output Evaluation
----------------------------------------------------------------
- Runs completions using **OpenAI GPT-3/4 (API) and local HuggingFace models**
- Evaluates both with the exact same BlazeMetrics API (text metrics, guardrails)
- Demonstrates blocklist/guardrail/safety triggering and metrics on both
- Shows example output for both providers
- Ready for integration with enterprise API-based LLMs

**Requirements:**
- Internet connection
- `OPENAI_API_KEY` set (for OpenAI)
- `transformers`, `torch` (for HuggingFace)
"""

import os
from blazemetrics import BlazeMetricsClient

# ----- OpenAI LLM API Completion -----
import openai

prompt_openai = "What is the boiling point of gold in Celsius?"
api_key = os.getenv("OPENAI_API_KEY")
openai_output = None

if api_key:
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",  # Always use latest GPT-4o
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_openai}
            ],
            max_tokens=32
        )
        openai_output = response.choices[0].message.content.strip()
    except Exception as ex:
        openai_output = f"OpenAI API error: {ex}"
else:
    openai_output = "API_KEY not set."

# ----- Local HuggingFace Text Generation -----
try:
    from transformers import pipeline
    # Use a small, local model for quick demonstration
    hf_pipe = pipeline("text-generation", model="distilgpt2")
    prompt_hf = "The capital of Brazil is"
    hf_result = hf_pipe(prompt_hf, max_new_tokens=12)[0]
    hf_output = hf_result.get("generated_text", hf_result.get("text", "")).strip()
except Exception as ex:
    hf_output = f"HuggingFace transformers error: {type(ex).__name__}: {ex}"

# ----- Unified BlazeMetrics Evaluation -----
# Guardrail: block dangerous/undesired words
blocklist = ["kill", "danger", "explode", "toxic"]

client = BlazeMetricsClient(
    blocklist=blocklist,
    redact_pii=True,
    enable_analytics=False,
    metrics_lowercase=True,   # Normalize for fair scoring
)

# References for metrics: use known truths/context
references = [
    ["Gold boils at 2856°C (2583 K)."],   # OpenAI completion reference
    ["The capital of Brazil is Brasília."], # HF completion reference
]

all_outputs = [openai_output, hf_output]
all_prompts = [prompt_openai, prompt_hf]

metrics = client.compute_metrics(
    all_outputs, references, include=client.config.metrics_include,
    lowercase=client.config.metrics_lowercase,
)
safety = client.check_safety(all_outputs)

# --------------- Output for both LLMs ----------------------
for idx, (provider, out, ref) in enumerate([
    ("OpenAI", openai_output, references[0][0]),
    ("HuggingFace", hf_output, references[1][0])
]):
    print(f"\nProvider: {provider}")
    print(f"Prompt:    {all_prompts[idx]!r}")
    print(f"Output:    {out!r}")
    print(f"Reference: {ref!r}")
    print("Metrics:")
    for metric, vals in metrics.items():
        print(f"  {metric}: {vals[idx]}")
    print("Safety/Guardrails:")
    print(safety[idx])
    print("-" * 48)

"""
Best practices illustrated:
- Providers: Switch between OpenAI API or HuggingFace local with zero changes to downstream pipeline
- Customizable guardrails for content safety: blocklist, PII redaction, regex, etc
- Reference-based BLEU/ROUGE/CHRF/token-F1/etc scoring
- Add more LLMs by appending completions and references
- Robust error-handling (provider unavailable)

For local HF, try swapping in larger models (e.g. 'tiiuae/falcon-7b'), or an instruction-tuned checkpoint.
"""