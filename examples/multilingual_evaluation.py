"""
Example: Multi-lingual Evaluation with BlazeMetrics

Demonstrates how to use BlazeMetrics for metrics and guardrails on non-English texts.
"""

from blazemetrics.client import BlazeMetricsClient

# Example: French and Spanish candidate/reference pairs
candidates = [
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "El zorro marrón rápido salta sobre el perro perezoso."
]
references = [
    ["Un renard brun rapide saute par-dessus un chien paresseux."],
    ["Un zorro marrón rápido salta sobre un perro perezoso."]
]

# Example: Multi-lingual blocklist
texts = [
    "Ceci est un message propre.",
    "Este texto contiene una palabra prohibida: malo."
]
blocklist = ["malo", "prohibida"]

client = BlazeMetricsClient()

# Compute metrics (ROUGE, BLEU, etc.) for non-English texts
metrics = client.compute_metrics(candidates, references, include=["rouge1", "bleu"])
print("Multi-lingual Metrics:", metrics)

# Guardrails: blocklist in Spanish
blocked = client.fuzzy_blocklist(texts, blocklist)
for i, (text, flag) in enumerate(zip(texts, blocked)):
    print(f"Text {i+1}: {text}")
    print("  Blocked:", flag)
    print()