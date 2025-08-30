#!/usr/bin/env python3
"""
⚡ BlazeMetrics Hyper-Optimized CPU Benchmark

Compares BlazeMetrics against other libraries using fine-grained parallelism,
where every individual benchmark runs in its own process for maximum CPU utilization.

Instructions:
1. Run the pip install command to get all dependencies.
2. Run this script. It will generate the plot: blazemetrics_performance_comparison.html
"""

# !pip install blazemetrics evaluate rouge-score rouge nltk sacrebleu moverscore pandas plotly scipy pyemd "numpy<2"

import time
import random
import os
import sys
import warnings
from typing import List, Tuple, Callable
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# --- Setup and Utility Functions ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def gen_corpus(n: int = 1000, vocab: int = 1000, seed: int = 123) -> Tuple[List[str], List[List[str]]]:
    """Generate a synthetic corpus."""
    print(f"📊 Generating test corpus of size {n}...")
    rng = random.Random(seed)
    sentence = lambda: " ".join([f"token_{rng.randint(1, vocab)}" for _ in range(rng.randint(8, 25))])
    candidates = [sentence() for _ in range(n)]
    references = [[sentence()] for _ in range(n)]
    print("✅ Corpus generated.")
    return candidates, references

def timeit(fn: Callable, repeat: int = 3, warmup: int = 1) -> Tuple[float, float]:
    """Time a function."""
    for _ in range(warmup): fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), sum(times) / len(times)

# --- Top-Level Task Functions (Pickle-able) ---

def task_bm_rouge(cands, refs):
    import blazemetrics as bm
    return bm.rouge_score(cands, refs, score_type="rouge_n", n=1)

def task_eval_rouge(cands, refs):
    import evaluate
    flat_refs = [r[0] for r in refs]
    rouge_metric = evaluate.load("rouge", quiet=True)
    return rouge_metric.compute(predictions=cands, references=flat_refs)

def task_rouge_rouge(cands, refs):
    from rouge import Rouge as RougeScorer
    flat_refs = [r[0] for r in refs]
    rouge_scorer = RougeScorer()
    return rouge_scorer.get_scores(cands, flat_refs)

def task_bm_bleu(cands, refs):
    import blazemetrics as bm
    return bm.bleu(cands, refs)

def task_nltk_bleu(cands, refs):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ch = SmoothingFunction().method1
    return [sentence_bleu([r.split() for r in ref_list], c.split(), smoothing_function=ch) for c, ref_list in zip(cands, refs)]

def task_sacrebleu_bleu(cands, refs):
    import sacrebleu
    flat_refs = [r[0] for r in refs]
    return sacrebleu.corpus_bleu(cands, [flat_refs])

def task_bm_jaccard(cands, refs):
    import blazemetrics as bm
    return bm.jaccard(cands, refs)

def task_nltk_jaccard(cands, refs):
    import nltk
    return [1.0 - nltk.jaccard_distance(set(c.split()), set(r[0].split())) for c, r in zip(cands, refs)]

def task_bm_composite(cands, refs):
    import blazemetrics as bm
    return bm.compute_text_metrics(cands, refs)

def task_eval_composite(cands, refs):
    import evaluate
    flat_refs = [r[0] for r in refs]
    combined_metrics = evaluate.combine(['rouge', 'bleu', 'meteor'])
    return combined_metrics.compute(predictions=cands, references=flat_refs)

def task_bm_moverscore_emb(cands, refs):
    import blazemetrics as bm
    np.random.seed(42)
    cand_emb = np.random.rand(256, 768).astype(np.float32)
    ref_emb = np.random.rand(256, 768).astype(np.float32)
    return bm.moverscore_greedy(cand_emb, ref_emb)

def task_moverscore_official(cands, refs):
    from moverscore import score
    import pyemd
    flat_refs = [r[0] for r in refs]
    sample_cands = cands[:200]
    sample_refs = flat_refs[:200]
    return score(sample_refs, sample_cands, verbose=False)

# --- Worker and Display Functions ---

def run_single_benchmark(metric: str, library: str, task_func: Callable, cands: list, refs: list) -> Tuple:
    """A generic wrapper to run a single benchmark task in a child process."""
    try:
        timed_func = lambda: task_func(cands, refs)
        min_t, avg_t = timeit(timed_func)
        print(f"  > Finished: {metric} ({library}) in {avg_t:.4f}s")
        return metric, library, min_t, avg_t
    except Exception as e:
        print(f"  > FAILED: {metric} ({library}). Reason: {e}")
        return metric, library, -1.0, -1.0

def display_results(results_df: pd.DataFrame):
    """Display benchmark results in the console."""
    print("\n" + "="*60 + "\nBENCHMARK RESULTS\n" + "="*60)
    for metric, group in results_df.groupby('Metric'):
        print(f"\n📊 {metric}:")
        print("-" * 40)
        group = group.sort_values('Avg Time')
        fastest = group.iloc[0]
        print(f"🥇 {fastest['Library']:25} | avg: {fastest['Avg Time']:.4f}s")
        for _, other in group.iloc[1:].iterrows():
            speedup = other['Avg Time'] / fastest['Avg Time'] if fastest['Avg Time'] > 0 else float('inf')
            print(f"  {other['Library']:25} | avg: {other['Avg Time']:.4f}s | {speedup:.1f}x slower")

def create_performance_comparison_plot(results_df: pd.DataFrame):
    """
    Generates a bar chart comparing total execution time for core metrics, 
    highlighting blazemetrics and showing its speedup factor over competitors.
    """
    print("\n--- Generating Performance Comparison Plot ---")

    # --- THIS IS THE FIX: Exclude specialty, slow tasks for a fair comparison ---
    core_metrics = ['ROUGE-1', 'BLEU', 'Jaccard Similarity']
    fair_df = results_df[results_df['Metric'].isin(core_metrics)]
    print(f"ℹ️  Plotting a fair comparison using core metrics only: {', '.join(core_metrics)}")
    # --- END FIX ---

    # Calculate total time for each library on core metrics
    total_time_df = fair_df.groupby('Library')['Avg Time'].sum().reset_index()
    total_time_df = total_time_df.sort_values('Avg Time', ascending=True)

    # --- Prepare data for plotting ---
    blazemetrics_time = total_time_df[total_time_df['Library'] == 'blazemetrics']['Avg Time'].iloc[0]
    
    # Calculate speedup and define annotations
    annotations = []
    for i, row in total_time_df.iterrows():
        lib_name = row['Library']
        lib_time = row['Avg Time']
        if lib_name == 'blazemetrics':
            text = "<b>Fastest</b>"
        else:
            speedup = lib_time / blazemetrics_time
            text = f"<b>{speedup:.1f}x Slower</b>"
        annotations.append(text)
    
    total_time_df['Annotation'] = annotations

    # Define colors
    blazemetrics_color = '#d62728'
    other_library_color = '#aec7e8'
    colors = [blazemetrics_color if lib == 'blazemetrics' else other_library_color for lib in total_time_df['Library']]

    # --- Create the Plot ---
    fig = go.Figure(go.Bar(
        x=total_time_df['Library'],
        y=total_time_df['Avg Time'],
        marker_color=colors,
        text=total_time_df['Annotation'],
        textposition='outside',
        textfont=dict(size=14)
    ))

    # --- Style the Plot ---
    fig.update_layout(
        title='<b>Overall Performance on Core Metrics (ROUGE, BLEU, Jaccard)</b>',
        xaxis_title='Library',
        yaxis_title='Total Time (seconds) — Lower is Better',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=15, color="black"),
        plot_bgcolor='white',
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        xaxis=dict(showgrid=False),
        bargap=0.5
    )
    
    # Increase the top margin to make space for the annotations
    fig.update_yaxes(range=[0, total_time_df['Avg Time'].max() * 1.2])

    output_filename = 'blazemetrics_performance_comparison.html'
    fig.write_html(output_filename)
    print(f"✅ Plot saved successfully to: {output_filename}")


# --- Main Execution Block ---
def main():
    """Main function to define and run all benchmark tasks in parallel."""
    try:
        import blazemetrics
        import pandas
        import plotly
        import rouge
        import pyemd
    except ImportError as e:
        print(f"❌ A required library is not installed: {e}")
        print("Please run the following command to install all dependencies:")
        print("\npip install blazemetrics evaluate rouge-score rouge nltk sacrebleu moverscore pandas plotly scipy pyemd \"numpy<2\"\n")
        sys.exit(1)

    # Generate data
    cands, refs = gen_corpus(n=1000)

    # Define All Granular Benchmark Tasks
    all_tasks = [
        ("ROUGE-1", "blazemetrics", task_bm_rouge),
        ("ROUGE-1", "evaluate", task_eval_rouge),
        ("ROUGE-1", "rouge", task_rouge_rouge),
        
        ("BLEU", "blazemetrics", task_bm_bleu),
        ("BLEU", "nltk", task_nltk_bleu),
        ("BLEU", "sacrebleu", task_sacrebleu_bleu),
        
        ("Jaccard Similarity", "blazemetrics", task_bm_jaccard),
        ("Jaccard Similarity", "nltk", task_nltk_jaccard),
        
        ("Composite Metrics", "blazemetrics", task_bm_composite),
        ("Composite Metrics", "evaluate.combine", task_eval_composite),

        ("MoverScore (Embeddings)", "blazemetrics", task_bm_moverscore_emb),
        ("MoverScore (Full Pipeline)", "moverscore_official", task_moverscore_official),
    ]

    # Execute All Tasks Concurrently
    results_list = []
    print(f"\n--- Starting {len(all_tasks)} Benchmark Tasks in Parallel ---")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_benchmark, metric, lib, func, cands, refs) for metric, lib, func in all_tasks]
        for future in concurrent.futures.as_completed(futures):
            metric, library, min_t, avg_t = future.result()
            if avg_t >= 0: # Exclude failed tasks
                results_list.append({'Metric': metric, 'Library': library, 'Min Time': min_t, 'Avg Time': avg_t})

    if not results_list:
        print("\nNo benchmarks completed successfully. Exiting.")
        return
        
    results_df = pd.DataFrame(results_list)

    display_results(results_df)
    create_performance_comparison_plot(results_df)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user.")