"""
Benchmarking: BlazeMetrics vs. Python Libraries (ROUGE, BLEU, WER)

Compares BlazeMetrics to sacrebleu, rouge-score, and jiwer on a large batch of candidate/reference pairs.
"""

import time
import random

from blazemetrics.client import BlazeMetricsClient

# Try to import competitors, install if missing
try:
    import sacrebleu
except ImportError:
    print("sacrebleu not installed. Install with 'pip install sacrebleu'")
    sacrebleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("rouge-score not installed. Install with 'pip install rouge-score'")
    rouge_scorer = None

try:
    import jiwer
except ImportError:
    print("jiwer not installed. Install with 'pip install jiwer'")
    jiwer = None

# Generate a large batch of synthetic data
N = 1000
candidates = [f"Sample candidate sentence {i}" for i in range(N)]
references = [[f"Sample reference sentence {i}"] for i in range(N)]

print("Benchmarking BlazeMetrics...")
client = BlazeMetricsClient()
start = time.time()
bm_metrics = client.compute_metrics(candidates, references, include=["rouge1", "bleu", "wer"])
bm_time = time.time() - start
print(f"BlazeMetrics time: {bm_time:.3f}s")

if sacrebleu:
    print("Benchmarking sacrebleu...")
    start = time.time()
    bleu_scores = [sacrebleu.sentence_bleu(c, [r[0]]).score for c, r in zip(candidates, references)]
    sacrebleu_time = time.time() - start
    print(f"sacrebleu time: {sacrebleu_time:.3f}s")
else:
    bleu_scores = None

if rouge_scorer:
    print("Benchmarking rouge-score...")
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    start = time.time()
    rouge1_scores = [scorer.score(c, r[0])['rouge1'].fmeasure for c, r in zip(candidates, references)]
    rouge_time = time.time() - start
    print(f"rouge-score time: {rouge_time:.3f}s")
else:
    rouge1_scores = None

if jiwer:
    print("Benchmarking jiwer (WER)...")
    start = time.time()
    wer_scores = [jiwer.wer(r[0], c) for c, r in zip(candidates, references)]
    jiwer_time = time.time() - start
    print(f"jiwer time: {jiwer_time:.3f}s")
else:
    wer_scores = None

print("\n--- Results ---")
print(f"BlazeMetrics ROUGE1: {bm_metrics['rouge1_f1'][:3]} ...")
if rouge1_scores is not None:
    print(f"rouge-score ROUGE1: {rouge1_scores[:3]} ...")
print(f"BlazeMetrics BLEU: {bm_metrics['bleu'][:3]} ...")
if bleu_scores is not None:
    print(f"sacrebleu BLEU: {bleu_scores[:3]} ...")
print(f"BlazeMetrics WER: {bm_metrics['wer'][:3]} ...")
if wer_scores is not None:
    print(f"jiwer WER: {wer_scores[:3]} ...")