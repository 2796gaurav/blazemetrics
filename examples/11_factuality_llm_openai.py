"""
11_factuality_llm_openai.py

BlazeMetrics Example – Real-World Factuality & Hallucination Judging
--------------------------------------------------------------------
- Uses a callable capable of LLM-based (OpenAI or HuggingFace) factuality evaluation
- Evaluates each output/reference for:
    - factuality (1: fully factual, 0: not factual)
    - hallucination (1: hallucinated, 0: factual)
    - detailed explanation/rationale

**Switch between OpenAI (API) or HuggingFace (local model) as needed.**
**Set OPENAI_API_KEY or use local HF model for private/faster/cheaper pipeline!**

----------------------------------------------------
"""

import os
from blazemetrics import BlazeMetricsClient

# ------------------- CONFIG -----------------------

JUDGE_BACKEND = os.getenv("FACT_JUDGE_BACKEND", "openai")   # "openai" or "huggingface"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_JUDGE_MODEL = os.getenv("BLAZEMETRICS_FAITH_HF_MODEL", "microsoft/deberta-v3-base-mnli") # Or your own model!
HF_DEVICE = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

# ------------- DATA: Demo includes multiple types -------------

outputs = [
    "Paris is the capital of France.",         # Factually true (factuality = 1.0)
    "Barack Obama was the 44th President.",    # True
    "The moon is made of cheese.",             # Hallucination
    "The sky is green during the day.",        # Hallucination/borderline
    "GPT-3 was created by OpenAI.",            # Fact
    "Bitcoin is a centralized bank.",          # Hallucination
]
references = [
    "Paris is the capital of France.",
    "Barack Obama served as the 44th US President.",
    "",
    "The sky appears blue during daylight.",
    "The GPT-3 language model was released by OpenAI.",
    "Bitcoin is a decentralized cryptocurrency.",
]

# -------------- OpenAI Judge Callable ---------------------
def openai_factuality_scorer(output, reference):
    import openai
    import json
    openai.api_key = OPENAI_API_KEY
    system = (
        "You are a trustworthy factuality judge. "
        "Rate the OUTPUT vs. REFERENCE for factuality (1.0: perfectly truthful to the reference, 0.0: entirely fabricated/hallucinated) "
        "and hallucination (1.0: complete hallucination, 0.0: none). "
        'Respond with JSON: {"factuality": float, "hallucination": float, "explanation": string}'
    )
    user = f"OUTPUT: {output}\nREFERENCE: {reference if reference else ''}"
    import openai as _openai
    client = _openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system},{"role": "user","content": user}],
        temperature=0.0, max_tokens=256)
    msg = resp.choices[0].message.content
    try:
        return json.loads(msg)
    except Exception:
        # Defensive: sometimes LLM output isn't strict JSON
        return {"factuality": 0.0, "hallucination": 1.0, "explanation": msg}

# ------------- HuggingFace Judge Callable ------------------
def huggingface_factuality_scorer(output, reference):
    """
    Simpler design: infers 'entailment' vs 'contradiction' using MNLI-style models.
    Should be run locally—NO usage costs. Installation: pip install transformers torch.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # Load on first call (static cache)
    if not hasattr(huggingface_factuality_scorer, "_model"):
        model = AutoModelForSequenceClassification.from_pretrained(HF_JUDGE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(HF_JUDGE_MODEL)
        model.eval()
        model.to(HF_DEVICE)
        huggingface_factuality_scorer._model = model
        huggingface_factuality_scorer._tokenizer = tokenizer
    model = huggingface_factuality_scorer._model
    tokenizer = huggingface_factuality_scorer._tokenizer
    # For entailment, premise is reference, hypothesis is output
    inputs = tokenizer(reference, output, return_tensors="pt", truncation=True).to(HF_DEVICE)
    with torch.no_grad():
        out = model(**inputs)
        scores = out.logits.softmax(dim=-1).cpu().numpy()[0]
    # MNLI label mapping
    label_map = model.config.id2label
    label = label_map[int(scores.argmax())]
    # Heuristic mapping
    factuality = float(scores[2])  # entailment
    hallucination = float(scores[0])  # contradiction
    explanation = f"HF MNLI label: {label}, entails: {factuality:.2f}, contradiction: {hallucination:.2f}"
    return {"factuality": factuality, "hallucination": hallucination, "explanation": explanation}

# ----------------- BlazeMetrics Pipeline ---------------------

client = BlazeMetricsClient()

if JUDGE_BACKEND.lower() == "huggingface":
    print("Using HuggingFace local judge model:", HF_JUDGE_MODEL)
    scorer = huggingface_factuality_scorer
else:
    print("Using OpenAI LLM judge (gpt-3.5/4; requires API KEY).")
    scorer = openai_factuality_scorer

client.set_factuality_scorer(scorer)
results = client.evaluate_factuality(outputs, references)

print(f"\n--- Factuality Results ({JUDGE_BACKEND}) ---")
for i, r in enumerate(results):
    print(f"{i+1}: Output: {outputs[i]!r}")
    print("  Reference:", references[i])
    print("  Factuality:", r.get("factuality"))
    print("  Hallucination:", r.get("hallucination"))
    print("  Explanation/Label:", r.get("explanation"))
    print()

"""
Tips:
- To use OpenAI: set OPENAI_API_KEY env var—this will call gpt-4o or gpt-4 for rating each output.
- To use a HuggingFace model: install transformers, torch, run with FACT_JUDGE_BACKEND=huggingface and (optionally) set BLAZEMETRICS_FAITH_HF_MODEL.
- Results may be slower but reproducible with HuggingFace, faster but more nuanced (and costlier) with OpenAI.
- For production, you can wrap either pipeline in async handlers, batch scoring, or policy conformities and reporting.
"""