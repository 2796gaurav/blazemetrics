#!/usr/bin/env python3
"""
⚡ BlazeMetrics Benchmarking & Comparative Analysis

This script compares BlazeMetrics against popular Python implementations where available.
- BLEU (`nltk`), chrF (`sacrebleu`), METEOR (`nltk`), WER (`jiwer`)
- Additional metrics: ROUGE (`evaluate`), text similarity (`rapidfuzz`), readability (`textstat`)
- Cleanly skips baselines when dependencies are missing
- Reports timings (min/avg over repeats) and performance comparisons
"""

import time
import random
import os
import json
import subprocess
import sys
import warnings
from typing import List, Dict, Any

# Suppress NumPy compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

# Function to install packages automatically
def install_package(package_name):
    """Install a package using pip if not already available"""
    try:
        # Skip import check for problematic packages
        if package_name in ["pandas", "nltk"]:
            print(f"Checking {package_name} availability...")
            return True
        
        __import__(package_name)
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return False

def gen_corpus(n=2000, vocab=1000, seed=123):
    """Generate synthetic corpus for benchmarking"""
    rng = random.Random(seed)
    
    def sentence(min_len=8, max_len=25):
        length = rng.randint(min_len, max_len)
        words = [f"token_{rng.randint(1, vocab)}" for _ in range(length)]
        return " ".join(words)
    
    candidates = [sentence() for _ in range(n)]
    references = [[sentence()] for _ in range(n)]
    return candidates, references

def timeit(fn, repeat=5, warmup=3):
    """Time a function with warmup and repeat"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), sum(times) / len(times)

def calculate_speedup(blazemetrics_time, baseline_time):
    """Calculate speedup factor"""
    if baseline_time > 0:
        return baseline_time / blazemetrics_time
    return float('inf')

def run_benchmarks():
    """Run all benchmarks and return results"""
    # Install required packages automatically
    print("🔧 Checking required packages...")
    install_package("pandas")
    install_package("nltk")
    install_package("sacrebleu")
    install_package("jiwer")
    install_package("evaluate")
    install_package("textstat")
    install_package("rapidfuzz")

    # Import BlazeMetrics functions
    from blazemetrics import (
        rouge_score as rg_rouge,
        bleu as rg_bleu,
        chrf_score as rg_chrf,
        meteor as rg_meteor,
        wer as rg_wer,
        bert_score_similarity as rg_bertsim,
        moverscore_greedy as rg_moverscore,
        compute_text_metrics as rg_compute_all,
        set_parallel,
        set_parallel_threshold,
    )
    
    # Configure parallelism settings
    set_parallel(True)
    set_parallel_threshold(100)

    # Optional baselines (best-effort imports)
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        print("✅ nltk imported successfully")
    except Exception as e:
        print(f"⚠️ nltk not available: {e}")
        nltk = None

    try:
        import sacrebleu
        print("✅ sacrebleu imported successfully")
    except Exception as e:
        print(f"⚠️ sacrebleu not available: {e}")
        sacrebleu = None

    try:
        import jiwer
        print("✅ jiwer imported successfully")
    except Exception as e:
        print(f"⚠️ jiwer not available: {e}")
        jiwer = None

    try:
        import evaluate
        print("✅ evaluate imported successfully")
    except Exception as e:
        print(f"⚠️ evaluate not available: {e}")
        evaluate = None

    try:
        import textstat
        print("✅ textstat imported successfully")
    except Exception as e:
        print(f"⚠️ textstat not available: {e}")
        textstat = None

    try:
        import rapidfuzz
        from rapidfuzz import fuzz
        print("✅ rapidfuzz imported successfully")
    except Exception as e:
        print(f"⚠️ rapidfuzz not available: {e}")
        rapidfuzz = None

    print("✅ Imports complete (missing baselines will be skipped)")

    # Generate test data
    print("📊 Generating test corpus...")
    cands, refs = gen_corpus(n=2000, vocab=1000)
    print(f"Generated corpus: {len(cands)} candidates, {len(refs)} references")

    # Embeddings for BERT similarity demo
    np.random.seed(42)
    cand_emb = np.random.rand(256, 768).astype(np.float32)
    ref_emb = np.random.rand(256, 768).astype(np.float32)

    results: Dict[str, list[tuple[str, float, float]]] = {}

    # ROUGE
    try:
        min_t, avg_t = timeit(lambda: rg_rouge(cands, refs, score_type="rouge_n", n=1))
        results.setdefault("ROUGE-1", []).append(("blazemetrics", min_t, avg_t))
        print("✅ ROUGE-1 blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ ROUGE-1 blazemetrics failed: {e}")

    if evaluate is not None:
        try:
            rouge = evaluate.load("rouge")
            def _evaluate_rouge():
                return rouge.compute(predictions=cands, references=[r[0] for r in refs])
            min_t, avg_t = timeit(_evaluate_rouge)
            results["ROUGE-1"].append(("evaluate", min_t, avg_t))
            print("✅ ROUGE-1 evaluate benchmarked")
        except Exception as e:
            print(f"❌ ROUGE-1 evaluate failed: {e}")

    # BLEU
    try:
        min_t, avg_t = timeit(lambda: rg_bleu(cands, refs))
        results.setdefault("BLEU", []).append(("blazemetrics", min_t, avg_t))
        print("✅ BLEU blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ BLEU blazemetrics failed: {e}")

    if nltk is not None:
        try:
            ch = SmoothingFunction().method1
            def _py_bleu():
                scores = []
                for c, rlist in zip(cands, refs):
                    ref_tokens = [r.split() for r in rlist]
                    cand_tokens = c.split()
                    scores.append(sentence_bleu(ref_tokens, cand_tokens, smoothing_function=ch))
                return scores
            min_t, avg_t = timeit(_py_bleu)
            results["BLEU"].append(("nltk", min_t, avg_t))
            print("✅ BLEU nltk benchmarked")
        except Exception as e:
            print(f"❌ BLEU nltk failed: {e}")

    if evaluate is not None:
        try:
            bleu = evaluate.load("bleu")
            def _evaluate_bleu():
                return bleu.compute(predictions=cands, references=[r[0] for r in refs])
            min_t, avg_t = timeit(_evaluate_bleu)
            results["BLEU"].append(("evaluate", min_t, avg_t))
            print("✅ BLEU evaluate benchmarked")
        except Exception as e:
            print(f"❌ BLEU evaluate failed: {e}")

    # chrF
    try:
        min_t, avg_t = timeit(lambda: rg_chrf(cands, refs))
        results.setdefault("chrF", []).append(("blazemetrics", min_t, avg_t))
        print("✅ chrF blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ chrF blazemetrics failed: {e}")

    if sacrebleu is not None:
        try:
            def _py_chrf():
                return [sacrebleu.corpus_chrf(cands, list(zip(*refs))[0]).score]  # type: ignore
            min_t, avg_t = timeit(_py_chrf)
            results["chrF"].append(("sacrebleu", min_t, avg_t))
            print("✅ chrF sacrebleu benchmarked")
        except Exception as e:
            print(f"❌ chrF sacrebleu failed: {e}")

    if evaluate is not None:
        try:
            chrf = evaluate.load("chrf")
            def _evaluate_chrf():
                return chrf.compute(predictions=cands, references=[r[0] for r in refs])
            min_t, avg_t = timeit(_evaluate_chrf)
            results["chrF"].append(("evaluate", min_t, avg_t))
            print("✅ chrF evaluate benchmarked")
        except Exception as e:
            print(f"❌ chrF evaluate failed: {e}")

    # METEOR
    try:
        min_t, avg_t = timeit(lambda: rg_meteor(cands, refs))
        results.setdefault("METEOR", []).append(("blazemetrics", min_t, avg_t))
        print("✅ METEOR blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ METEOR blazemetrics failed: {e}")

    if nltk is not None:
        try:
            def _py_meteor():
                return [nltk_meteor(r[0], c) for c, r in zip(cands, refs)]
            min_t, avg_t = timeit(_py_meteor)
            results["METEOR"].append(("nltk", min_t, avg_t))
            print("✅ METEOR nltk benchmarked")
        except Exception as e:
            print(f"❌ METEOR nltk failed: {e}")

    # WER
    try:
        min_t, avg_t = timeit(lambda: rg_wer(cands, refs))
        results.setdefault("WER", []).append(("blazemetrics", min_t, avg_t))
        print("✅ WER blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ WER blazemetrics failed: {e}")

    if jiwer is not None:
        try:
            def _py_wer():
                return [jiwer.wer(r[0], c) for c, r in zip(cands, refs)]
            min_t, avg_t = timeit(_py_wer)
            results["WER"].append(("jiwer", min_t, avg_t))
            print("✅ WER jiwer benchmarked")
        except Exception as e:
            print(f"❌ WER jiwer failed: {e}")

    # Text Similarity (RapidFuzz)
    if rapidfuzz is not None:
        try:
            def _rapidfuzz_similarity():
                return [fuzz.ratio(c, r[0]) for c, r in zip(cands, refs)]
            min_t, avg_t = timeit(_rapidfuzz_similarity)
            results.setdefault("Text Similarity", []).append(("rapidfuzz", min_t, avg_t))
            print("✅ Text Similarity rapidfuzz benchmarked")
        except Exception as e:
            print(f"❌ Text Similarity rapidfuzz failed: {e}")

    # Readability (TextStat)
    if textstat is not None:
        try:
            def _textstat_readability():
                return [textstat.flesch_reading_ease(c) for c in cands]
            min_t, avg_t = timeit(_textstat_readability)
            results.setdefault("Readability", []).append(("textstat", min_t, avg_t))
            print("✅ Readability textstat benchmarked")
        except Exception as e:
            print(f"❌ Readability textstat failed: {e}")

    # BERT similarity
    try:
        min_t, avg_t = timeit(lambda: rg_bertsim(cand_emb, ref_emb))
        results.setdefault("BERT-sim", []).append(("blazemetrics", min_t, avg_t))
        print("✅ BERT-sim blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ BERT-sim blazemetrics failed: {e}")

    # MoverScore
    try:
        min_t, avg_t = timeit(lambda: rg_moverscore(cand_emb, ref_emb))
        results.setdefault("MoverScore (greedy)", []).append(("blazemetrics", min_t, avg_t))
        print("✅ MoverScore (greedy) blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ MoverScore (greedy) blazemetrics failed: {e}")

    # Comprehensive metrics
    try:
        min_t, avg_t = timeit(lambda: rg_compute_all(cands, refs))
        results.setdefault("Comprehensive", []).append(("blazemetrics", min_t, avg_t))
        print("✅ Comprehensive metrics blazemetrics benchmarked")
    except Exception as e:
        print(f"❌ Comprehensive metrics blazemetrics failed: {e}")

    print("✅ Benchmarks complete")
    print(f"Results: {len(results)} metrics benchmarked")
    for metric, entries in results.items():
        print(f"  {metric}: {len(entries)} implementations")
    
    return results

def display_results(results):
    """Display benchmark results with speedup calculations"""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for metric, entries in results.items():
        print(f"\n📊 {metric}:")
        print("-" * 40)
        
        # Sort by average time (fastest first)
        entries.sort(key=lambda x: x[2])
        
        for i, (name, tmin, tavg) in enumerate(entries):
            if i == 0:
                print(f"🥇 {name:15} | min: {tmin:.4f}s | avg: {tavg:.4f}s")
            else:
                # Calculate speedup compared to fastest
                fastest_time = entries[0][2]
                speedup = calculate_speedup(fastest_time, tavg)
                print(f"   {name:15} | min: {tmin:.4f}s | avg: {tavg:.4f}s | {speedup:.1f}x slower")
        
        # Show speedup if there are multiple implementations
        if len(entries) > 1:
            fastest = entries[0]
            slowest = entries[-1]
            if fastest[0] == "blazemetrics":
                speedup = calculate_speedup(fastest[2], slowest[2])
                print(f"   🚀 BlazeMetrics is {speedup:.1f}x faster than {slowest[0]}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_metrics = len(results)
    blazemetrics_wins = 0
    blazemetrics_only = 0
    
    for metric, entries in results.items():
        if len(entries) == 1 and entries[0][0] == "blazemetrics":
            blazemetrics_only += 1
        elif entries[0][0] == "blazemetrics":
            blazemetrics_wins += 1
    
    print(f"📈 Total metrics tested: {total_metrics}")
    print(f"🏆 BlazeMetrics wins: {blazemetrics_wins}")
    print(f"👑 BlazeMetrics only: {blazemetrics_only}")
    print(f"🎯 BlazeMetrics success rate: {((blazemetrics_wins + blazemetrics_only) / total_metrics * 100):.1f}%")

def main():
    """Main function to run benchmarks"""
    try:
        results = run_benchmarks()
        display_results(results)
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



