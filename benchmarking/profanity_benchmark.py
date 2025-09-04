"""
Benchmarking: BlazeMetrics Profanity Detection vs. profanity-check and profanity-filter

Compares BlazeMetrics' blocklist/fuzzy guardrails to profanity-check and profanity-filter on large batches.
"""

import time
from blazemetrics.client import BlazeMetricsClient

try:
    from profanity_check import predict as pc_predict
except ImportError:
    print("profanity-check not installed. Install with 'pip install profanity-check'")
    pc_predict = None

try:
    from profanity_filter import ProfanityFilter
except ImportError:
    print("profanity-filter not installed. Install with 'pip install profanity-filter'")
    ProfanityFilter = None

# Generate a large batch of texts (some clean, some profane)
N = 1000
texts = [
    "This is a clean message." if i % 3 == 0 else
    "You are an idiot!" if i % 3 == 1 else
    "This contains a bаdword (Cyrillic 'а')."
    for i in range(N)
]
patterns = ["idiot", "badword"]

client = BlazeMetricsClient(blocklist=patterns)

print("Benchmarking BlazeMetrics fuzzy_blocklist...")
start = time.time()
bm_results = client.fuzzy_blocklist(texts, patterns)
bm_time = time.time() - start
print(f"BlazeMetrics fuzzy_blocklist time: {bm_time:.3f}s")
print("Sample BlazeMetrics results:", bm_results[:10])

if pc_predict:
    print("Benchmarking profanity-check...")
    start = time.time()
    pc_results = pc_predict(texts)
    pc_time = time.time() - start
    print(f"profanity-check time: {pc_time:.3f}s")
    print("Sample profanity-check results:", pc_results[:10])

if ProfanityFilter:
    print("Benchmarking profanity-filter...")
    pf = ProfanityFilter()
    start = time.time()
    pf_results = [pf.is_profane(text) for text in texts]
    pf_time = time.time() - start
    print(f"profanity-filter time: {pf_time:.3f}s")
    print("Sample profanity-filter results:", pf_results[:10])