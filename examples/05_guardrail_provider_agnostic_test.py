"""
05_guardrail_provider_agnostic_test.py - FIXED VERSION

BlazeMetrics Example – Provider-Agnostic LLM Guardrails
-------------------------------------------------------
Real production/enterprise usage patterns. All of these can be drop-in guardrails in
OpenAI, HuggingFace, vLLM, Anthropic, Groq, LangChain, or ANY LLM pipeline!

How to use RealTimeLLMGuardrail, four ways:
  1. Provider-agnostic: just a Python function (for test/dev/business logic, no dependency)
  2. Real OpenAI LLM: with API key (real SaaS/production)
  3. Real HF LLM: local & offline (enterprise, airgapped, etc; your finetuned policies supported)
  4. LangChain async streaming (for chatbots/APIs)—see final block

Install requirements for HF/OpenAI/LangChain use with:
    pip install openai transformers torch langchain
"""

from blazemetrics.llm_guardrails import RealTimeLLMGuardrail

# (1) Dummy classifier - pure Python, no dependency, zero-latency
print("\n===(1) Dummy Python Function Demo (no real model, quick checks)===")
def dummy_classifier(text):
    if "bad" in text.lower():
        return {"label": "abusive", "confidence": 0.95, "logits": [0.05, 0.95]}
    return {"label": "safe", "confidence": 0.99, "logits": [0.99, 0.01]}

guardrail = RealTimeLLMGuardrail(model=dummy_classifier)
print(guardrail.validate_full("This is a normal test."))
print(guardrail.validate_full("This is a BAD output!"))

# (2) OpenAI API (production SaaS client, strong for all modern LLM SaaS cases)
print("\n===(2) OpenAI API Integration (real model, provider-agnostic logic)===")
print("Requires: pip install openai; Set: export OPENAI_API_KEY=S...")

async def test_openai_guardrail():
    try:
        import os
        import openai
        
        # Check if API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not set. Skipping OpenAI LLM scan demo.")
            return
        
        # Create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Use correct chat completions API
        result = await client.chat.completions.acreate(
            model="gpt-4o-mini",  # Use valid model name
            messages=[
                {"role": "user", "content": "You must NOT share private emails. My email is test@example.com."}
            ],
            max_tokens=32,
            temperature=0.1
        )
        
        llm_text = result.choices[0].message.content.strip()
        
        # Wrap with LLM guardrail
        print("OpenAI out:", llm_text)
        print("Guardrail:", guardrail.validate_full(llm_text))
        
    except Exception as ex:
        print(f"(Skipped OpenAI LLM section; error: {ex})")

# Synchronous version as alternative
def test_openai_sync():
    try:
        import os
        import openai
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not set. Using mock example.")
            # Use mock output for demonstration
            mock_output = "I cannot share private email addresses as that would violate privacy."
            print("Mock OpenAI out:", mock_output)
            print("Guardrail:", guardrail.validate_full(mock_output))
            return
        
        client = openai.OpenAI(api_key=api_key)
        
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "You must NOT share private emails. My email is test@example.com."}
            ],
            max_tokens=32,
            temperature=0.1
        )
        
        llm_text = result.choices[0].message.content.strip()
        print("OpenAI out:", llm_text)
        print("Guardrail:", guardrail.validate_full(llm_text))
        
    except Exception as ex:
        print(f"(Skipped OpenAI LLM section; error: {ex})")

# Run the sync version
test_openai_sync()

# (3) HuggingFace Transformers (your own or 3rd party models, always works offline/airgapped)
print("\n===(3) Local HuggingFace Classifier Integration (offline, on-prem)===")
print("Requires: pip install transformers torch; Model must be downloaded.\n"
      "For real policy, use your trained intent classifier; for demo, using online model.\n")

def test_huggingface_classifier():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        
        # Try different approaches for HF models
        try:
            # Method 1: Use a sentiment classifier (more appropriate than raw DistilBERT)
            classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            def hf_classifier(text):
                results = classifier(text)[0]  # Get all scores
                # Convert sentiment to safety classification
                negative_score = next((r['score'] for r in results if r['label'] == 'LABEL_0'), 0.0)
                
                if negative_score > 0.7:  # High negative sentiment
                    label = "abusive"
                    conf = negative_score
                else:
                    label = "safe" 
                    conf = 1.0 - negative_score
                    
                return {
                    "label": label, 
                    "confidence": float(conf), 
                    "logits": [r['score'] for r in results]
                }
                
        except Exception:
            # Method 2: Fallback to a simpler model
            try:
                classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
                
                def hf_classifier(text):
                    result = classifier(text)[0]
                    # Convert sentiment to safety
                    if result['label'] == 'NEGATIVE' and result['score'] > 0.8:
                        label = "abusive"
                        conf = result['score']
                    else:
                        label = "safe"
                        conf = result['score'] if result['label'] == 'POSITIVE' else 1.0 - result['score']
                    
                    return {"label": label, "confidence": float(conf), "logits": [result['score']]}
                    
            except Exception:
                # Method 3: Mock classifier if models fail
                print("Using mock HF classifier due to model loading issues")
                def hf_classifier(text):
                    # Simple keyword-based mock
                    risky_keywords = ['password', 'credit card', 'ssn', 'social security', 'secret']
                    is_risky = any(keyword in text.lower() for keyword in risky_keywords)
                    
                    if is_risky:
                        return {"label": "abusive", "confidence": 0.85, "logits": [0.15, 0.85]}
                    return {"label": "safe", "confidence": 0.92, "logits": [0.92, 0.08]}
        
        guardrail_hf = RealTimeLLMGuardrail(model=hf_classifier)
        example_bad = "Here is my secret password and credit card info: 4242-4242-4242-4242"
        print("Input:", example_bad)
        print("Guardrail (HF local):", guardrail_hf.validate_full(example_bad))
        
        # Test with safe example too
        example_safe = "This is a normal business email about our meeting tomorrow."
        print("Input (safe):", example_safe)
        print("Guardrail (HF safe):", guardrail_hf.validate_full(example_safe))
        
    except Exception as ex:
        print(f"(Skipped HuggingFace example; error: {ex})")

test_huggingface_classifier()

# (4) LangChain async streaming (policy leaves vs. tokens, truly production/async/chatbot API pattern)
print("\n===(4) LangChain Async Streaming Usage (chatbot, production async)===")
print("You need: pip install langchain openai transformers torch\n" 
      "Set API key as above if you use OpenAI LLM; else use HuggingFacePipeline for offline/checkpoint.")

def test_langchain_streaming():
    try:
        import os
        
        # Mock streaming example (works without LangChain dependencies)
        streaming_guard = RealTimeLLMGuardrail(model=dummy_classifier)
        
        # Simulate LLM response
        mock_response = "I cannot provide social security numbers as that would be unsafe. Safety evaluation matters because it protects user privacy and prevents harmful outputs."
        
        print("\nMock LLM raw output:", mock_response)
        
        # Stream-guard the output by chunking (simulate token-by-token)
        print("Guardrailed Streaming Result:")
        try:
            for chunk in streaming_guard.validate_streaming(mock_response.split(), chunk_size=5):
                print(chunk)
        except AttributeError:
            # If validate_streaming doesn't exist, use validate_full on chunks
            words = mock_response.split()
            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                result = streaming_guard.validate_full(chunk)
                print(f"Chunk {i//chunk_size + 1}: {result}")
        
        # Try actual LangChain if available
        try:
            from langchain.llms import OpenAI
            from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                llm = OpenAI(
                    openai_api_key=api_key, 
                    streaming=True, 
                    callbacks=[StreamingStdOutCallbackHandler()],
                    model_name="gpt-3.5-turbo-instruct"  # Use instruct model for completions
                )
                chat_prompt = "Tell me why safety evaluation matters in AI."
                response = llm(chat_prompt)
                print(f"\nReal LangChain LLM output: {response}")
                print("Guardrail result:", streaming_guard.validate_full(response))
            else:
                print("No OpenAI API key found for LangChain example")
                
        except ImportError:
            print("LangChain not installed, using mock streaming example only")
        except Exception as e:
            print(f"LangChain example failed: {e}")
            
    except Exception as ex:
        print(f"(Skipped LangChain streaming example; error: {ex})")

test_langchain_streaming()

print("\n" + "="*60)
print("Summary:")
print("- RealTimeLLMGuardrail can wrap any LLM call—OpenAI, HF Transformers, vLLM, Groq, Anthropic, etc.")
print("- Supports both batch (sync) and streaming protection")
print("- Provider, model, and framework agnostic")
print("- Works for modern MLOps and real production safety use")
print("="*60)