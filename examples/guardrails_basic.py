"""
Example: Basic Guardrails with BlazeMetrics

This script demonstrates how to use blocklist, regex, and PII guardrails with BlazeMetrics.
"""

from blazemetrics.client import BlazeMetricsClient

texts = [
    "My credit card number is 4111-1111-1111-1111.",
    "This contains a badword and a suspicious pattern: DROP TABLE users;"
]

client = BlazeMetricsClient()

# Blocklist check
blocklist_result = client.fuzzy_blocklist(texts, patterns=["badword", "DROP TABLE"])
print("Blocklist:", blocklist_result)

# PII detection
pii_result = client.detect_pii(texts)
print("PII Detection:", pii_result)

# Regex guardrail (e.g., SQL injection pattern)
regex_result = client.check_safety(texts)
print("Safety Check:", regex_result)