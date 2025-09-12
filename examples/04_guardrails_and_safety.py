"""
04_guardrails_and_safety.py - FIXED VERSION (HuggingFace Section)

BlazeMetrics Example – Guardrails and Safety Checks with HuggingFace
"""
from blazemetrics import BlazeMetricsClient

# Basic guardrails setup
texts = [
    "Confidential: My email is alice@example.com.",
    "Hate speech: KKK propaganda.",
    "Harmless text for testing.",
]

blocklist = ["KKK", "hate"]
regexes = [r"\bemail\b"]

client = BlazeMetricsClient(blocklist=blocklist, regexes=regexes)

# Method 1: Use a proper text generation model
def test_with_gpt2():
    """Using GPT-2 which is available and works for text generation"""
    try:
        from transformers import pipeline
        
        # Use GPT-2 for text generation (small model, good for testing)
        hf_pipe = pipeline(
            "text-generation",
            model="gpt2",
            tokenizer="gpt2",
        )
        
        prompt = "Write an email sharing my credit card: 4242-4242-4242-4242"
        response = hf_pipe(
            prompt, 
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256  # GPT-2's EOS token
        )
        
        hf_output = response[0]["generated_text"]
        safety = client.check_safety([hf_output])[0]
        
        print("\n[HuggingFace GPT-2 Output with Guardrails]")
        print("  Output:", hf_output)
        print("  Blocked:", safety.get("blocked"))
        print("  Regex Flagged:", safety.get("regex_flagged"))
        print("  Redacted:", safety.get("redacted"))
        print("  Safety Score:", safety.get("safety_score"))
        print("  Final Output:", safety.get("final_output"))
        
    except Exception as e:
        print(f"Error with GPT-2: {e}")

# Method 2: Use DistilGPT-2 (smaller, faster alternative)
def test_with_distilgpt2():
    """Using DistilGPT-2 which is smaller and faster"""
    try:
        from transformers import pipeline
        
        hf_pipe = pipeline(
            "text-generation",
            model="distilgpt2",
            tokenizer="distilgpt2",
        )
        
        prompt = "My personal information includes email:"
        response = hf_pipe(
            prompt, 
            max_new_tokens=25,
            do_sample=True,
            temperature=0.8,
            pad_token_id=50256
        )
        
        hf_output = response[0]["generated_text"]
        safety = client.check_safety([hf_output])[0]
        
        print("\n[HuggingFace DistilGPT-2 Output with Guardrails]")
        print("  Output:", hf_output)
        print("  Blocked:", safety.get("blocked"))
        print("  Regex Flagged:", safety.get("regex_flagged"))
        print("  Redacted:", safety.get("redacted"))
        print("  Safety Score:", safety.get("safety_score"))
        print("  Final Output:", safety.get("final_output"))
        
    except Exception as e:
        print(f"Error with DistilGPT-2: {e}")

# Method 3: Manual model loading (more control)
def test_with_manual_loading():
    """Load model manually for more control"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        model_name = "microsoft/DialoGPT-small"  # Good for conversational text
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Use CPU
        )
        
        prompt = "Tell me your credit card number:"
        response = hf_pipe(
            prompt,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        
        hf_output = response[0]["generated_text"]
        safety = client.check_safety([hf_output])[0]
        
        print("\n[HuggingFace DialoGPT Output with Guardrails]")
        print("  Output:", hf_output)
        print("  Blocked:", safety.get("blocked"))
        print("  Regex Flagged:", safety.get("regex_flagged"))
        print("  Redacted:", safety.get("redacted"))
        print("  Safety Score:", safety.get("safety_score"))
        print("  Final Output:", safety.get("final_output"))
        
    except Exception as e:
        print(f"Error with manual loading: {e}")

# Method 4: Simple mock example if models don't work
def test_with_mock_output():
    """Test guardrails with mock LLM output"""
    mock_outputs = [
        "Sure! My credit card number is 4242-4242-4242-4242 and my email is user@example.com",
        "I cannot share sensitive financial information.",
        "Here's some harmless generated text for testing."
    ]
    
    print("\n[Mock LLM Outputs with Guardrails]")
    for i, output in enumerate(mock_outputs):
        safety = client.check_safety([output])[0]
        print(f"\nMock Output #{i+1}: {output}")
        print("  Blocked:", safety.get("blocked"))
        print("  Regex Flagged:", safety.get("regex_flagged"))
        print("  Redacted:", safety.get("redacted"))
        print("  Safety Score:", safety.get("safety_score"))
        print("  Final Output:", safety.get("final_output"))

if __name__ == "__main__":
    print("Testing HuggingFace models with BlazeMetrics guardrails...")
    
    # Try different approaches
    test_with_gpt2()
    test_with_distilgpt2()
    test_with_manual_loading()
    test_with_mock_output()
    
    print("\nNote: If models fail to download, they'll be cached for future runs.")
    print("You can also test with mock outputs to see how guardrails work.")