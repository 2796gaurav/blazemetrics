"""
Example: Custom PII Patterns with BlazeMetrics

Demonstrates how to add and use custom PII patterns for redaction and detection.
"""

from blazemetrics.llm_integrations import EnhancedPIIDetector

# Define custom PII patterns (e.g., API keys, custom tokens)
custom_patterns = {
    "api_key": [r"sk-[A-Za-z0-9]{32,}"],
    "custom_token": [r"TOKEN_[A-Z0-9]{10,}"]
}

texts = [
    "Here is my API key: sk-1234567890abcdef1234567890abcdef",
    "This is a custom token: TOKEN_ABCDEFGH1234",
    "No sensitive info here."
]

detector = EnhancedPIIDetector(custom_patterns=custom_patterns)
results = [detector.detect_pii(text) for text in texts]

for i, (text, res) in enumerate(zip(texts, results)):
    print(f"Text {i+1}: {text}")
    print("  Redacted:", res.redacted_text)
    print("  Detected Types:", res.detected_types)
    print("  Redaction Count:", res.redaction_count)
    print()