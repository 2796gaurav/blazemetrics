"""
Example: Fuzzy Blocklist & Unicode Spoofing Detection with BlazeMetrics

Demonstrates how to block obfuscated and Unicode-variant bad words using BlazeMetrics' fuzzy and Unicode-aware guardrails.
"""

from blazemetrics.client import BlazeMetricsClient

texts = [
    "This is a clean message.",
    "This contains a bаdword (Cyrillic 'а').",
    "Obfuscated: b.a.d.w.o.r.d",
    "Obfuscated: b a d w o r d",
    "Unicode: bаdwоrd (Cyrillic 'а' and 'о')",
    "Normal: nothing to see here."
]

patterns = ["badword"]

client = BlazeMetricsClient()

# Fuzzy blocklist check
fuzzy_results = client.fuzzy_blocklist(texts, patterns)
for i, (text, flagged) in enumerate(zip(texts, fuzzy_results)):
    print(f"Text {i+1}: {text}")
    print("  Fuzzy Blocked:", flagged)
    print()