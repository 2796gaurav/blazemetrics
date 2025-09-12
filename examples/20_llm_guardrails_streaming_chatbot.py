"""
20_llm_guardrails_streaming_chatbot.py

BlazeMetrics Example – Streaming LLM Moderation Chatbot Demo
-----------------------------------------------------------
Demonstrates real-time moderation/guardrails within a chatbot loop using BlazeMetrics:
- Simulates streaming token-by-token moderation for each user message
- Blocks or replaces outputs with a standard response if unsafe (abusive or "bad/hack" flagged)
- Good reference for integrating BlazeMetrics guardrails in interactive or chat-style apps

You can type any input; any string containing 'bad' or 'hack' is replaced by policy.
Type 'exit' to end example.
"""
from blazemetrics.llm_guardrails import RealTimeLLMGuardrail

def dummy_classifier(text):
    """Blocks everything with 'bad' or 'hack'."""
    if "bad" in text.lower() or "hack" in text.lower():
        return {"label": "abusive", "confidence": 0.9}
    return {"label": "safe", "confidence": 0.98}

guard = RealTimeLLMGuardrail(model=dummy_classifier)

print("=== Streaming LLM Chatbot Moderation ===")
while True:
    message = input("User> ")
    # Simulate token stream (split by space)
    stream = guard.validate_streaming(iter(message.split()), chunk_size=3)
    outputs = list(stream)
    print("Bot moderated:", " ".join(outputs))
    if outputs and outputs[0] == guard.standard_response:
        print("[MODERATION]: Message replaced by guardrail.")
    print()
    if "exit" in message:
        break
