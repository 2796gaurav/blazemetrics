import time
import string
import json
import psutil
import os
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from blazemetrics import BlazeMetricsClient
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# pip install psutil plotly evaluate sacrebleu nltk rouge-score
# Pre-load all libraries to avoid loading overhead in benchmarks
print(" Pre-loading libraries to ensure fair comparison...")

# Use lazy imports with caching
_evaluate_cache = None
_sacrebleu_cache = None
_nltk_cache = None
_rouge_scorer_cache = None
rouge = None
bleu = None
meteor = None

def get_evaluate():
    global _evaluate_cache, rouge, bleu, meteor
    if _evaluate_cache is None:
        try:
            import evaluate
            _evaluate_cache = evaluate
            
            # Pre-load evaluate models here to prevent NameError
            try:
                rouge = evaluate.load("rouge")
                bleu = evaluate.load("bleu")
                meteor = evaluate.load("meteor")
                print("    HuggingFace Evaluate models pre-loaded")
            except Exception as e:
                print(f"    Failed to pre-load HuggingFace Evaluate models: {e}")
                rouge = None
                bleu = None
                meteor = None
            
            print("    HuggingFace Evaluate loaded")
        except ImportError:
            _evaluate_cache = False
            print("    HuggingFace Evaluate not available")
    return _evaluate_cache if _evaluate_cache is not False else None

def get_sacrebleu():
    global _sacrebleu_cache
    if _sacrebleu_cache is None:
        try:
            import sacrebleu
            _sacrebleu_cache = sacrebleu
            print("    SacreBLEU loaded")
        except ImportError:
            _sacrebleu_cache = False
            print("    SacreBLEU not available")
    return _sacrebleu_cache if _sacrebleu_cache is not False else None

def get_nltk():
    global _nltk_cache
    if _nltk_cache is None:
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            from nltk.translate.bleu_score import sentence_bleu
            
            # Check if data is already downloaded, skip if present
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                # Only download if not present
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            
            _nltk_cache = {'nltk': nltk, 'meteor_score': meteor_score, 'sentence_bleu': sentence_bleu}
            print("    NLTK loaded and data downloaded")
        except ImportError:
            _nltk_cache = False
            print("    NLTK not available")
    return _nltk_cache if _nltk_cache is not False else None

def get_rouge_scorer():
    global _rouge_scorer_cache
    if _rouge_scorer_cache is None:
        try:
            from rouge_score import rouge_scorer
            _rouge_scorer_cache = rouge_scorer
            print("    rouge_score loaded")
        except ImportError:
            _rouge_scorer_cache = False
            print("    rouge_score not available")
    return _rouge_scorer_cache if _rouge_scorer_cache is not False else None

# Initialize caches
evaluate = get_evaluate()
sacrebleu = get_sacrebleu()
nltk_modules = get_nltk()
rouge_scorer = get_rouge_scorer()

HAS_ROUGE_SCORE = rouge_scorer is not None
HAS_BERTSCORE = False

# Skip BertScore by default as it's very slow
ENABLE_BERTSCORE = os.getenv('ENABLE_BERTSCORE', '0') == '1'

if ENABLE_BERTSCORE:
    try:
        from bert_score import score as bertscore_score
        HAS_BERTSCORE = True
        print("    bert_score loaded")
    except ImportError:
        bertscore_score = None
        print("    bert_score not available")

print(" All libraries pre-loaded. Starting benchmarks...\n")

# ==== Resource Monitoring ====

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        
    def start_monitoring(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def get_resource_usage(self):
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return {
            'memory_used_mb': current_memory - self.start_memory if self.start_memory else 0,
            'cpu_percent': self.process.cpu_percent()
        }

# ==== Optimized Data Generation ====

def normalize_text(s):
    """Optimized normalize text by lowercasing and removing punctuation"""
    if not hasattr(normalize_text, '_table'):
        normalize_text._table = str.maketrans('', '', string.punctuation)
    return s.lower().translate(normalize_text._table)

def normalize_batch(batch):
    """Vectorized normalization"""
    return [normalize_text(x) for x in batch]

def normalize_references(ref_batch):
    return [normalize_batch(refs) for refs in ref_batch]

def make_large_batch(size=1000):
    """Optimized batch generation using list comprehension"""
    base_text = "The quick brown fox jumps over the lazy dog number"
    candidates = [f"{base_text} {i}" for i in range(size)]
    references = [
        [f"A quick brown fox jumps over the lazy dog number {i}", 
         f"The fast brown fox jumps over the lazy dog number {i}"]
        for i in range(size)
    ]
    return candidates, references

def get_test_data(batch_size=2, normalize=True):
    """Optimized test data generation with caching"""
    if not hasattr(get_test_data, '_cache'):
        get_test_data._cache = {}
    
    cache_key = (batch_size, normalize)
    if cache_key in get_test_data._cache:
        return get_test_data._cache[cache_key]
    
    if batch_size > 2:
        candidates, references = make_large_batch(batch_size)
    else:
        candidates = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over the sleeping dog"
        ]
        references = [
            ["The brown fox jumped over the dog", "A quick fox jumps over a lazy dog"],
            ["A swift fox jumps over a resting dog", "The fast fox leaped over the dog"]
        ]
    
    if normalize:
        candidates = normalize_batch(candidates)
        references = normalize_references(references)
    
    result = (candidates, references)
    get_test_data._cache[cache_key] = result
    return result

# Optimized PreparedData class with pre-computed structures
class PreparedData:
    def __init__(self, batch_size: int, normalize_data: bool):
        self.candidates, self.references = get_test_data(batch_size, normalize=normalize_data)
        self.refs_transposed = list(zip(*self.references))
        self.refs_primary = [refs[0] for refs in self.references]
        
        # Pre-compute NLTK tokens only if NLTK is available
        self.nltk_refs_tokens = None
        self.nltk_cand_tokens = None
        if nltk_modules:
            self.nltk_refs_tokens = [[ref.split() for ref in refs] for refs in self.references]
            self.nltk_cand_tokens = [cand.split() for cand in self.candidates]
        
        # Pre-create rouge scorer only if available
        self.rouge_scorer_obj = None
        if HAS_ROUGE_SCORE:
            self.rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

# ==== Simplified Warm-up ====

def minimal_warmup():
    """Minimal warm-up that only tests availability without full computation"""
    print(" Quick warm-up...")
    
    # Quick test with tiny data
    tiny_candidates = ["test"]
    tiny_references = [["reference"]]
    
    try:
        client = BlazeMetricsClient()
        _ = client.compute_metrics(
            tiny_candidates, tiny_references,
            include=["rouge1"],  # Only one metric for speed
            lowercase=False,
            stemming=False
        )
        print("    BlazeMetrics ready")
    except Exception:
        print("    BlazeMetrics failed")
    
    # Skip other warmups for speed - they'll be tested in actual runs
    print(" Quick warm-up complete")

# ==== Optimized Benchmarks ====

def benchmark_blazemetrics_text_metrics(batch_size=2, normalize_data=True, prepared: Optional[PreparedData]=None):
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    client = BlazeMetricsClient()
    if prepared is None:
        candidates, references = get_test_data(batch_size, normalize=normalize_data)
    else:
        candidates, references = prepared.candidates, prepared.references
    
    start_time = time.perf_counter()
    results = client.compute_metrics(
        candidates, references,
        include=["rouge1", "rouge2", "rougeL", "bleu", "meteor", "wer", "token_f1", "jaccard"],
        lowercase=not normalize_data,
        stemming=False
    )
    end_time = time.perf_counter()
    
    aggs = client.aggregate_metrics(results)
    resources = monitor.get_resource_usage()
    
    return {
        'per_sample': results,
        'aggregate': aggs,
        'execution_time': end_time - start_time,
        'package': 'BlazeMetrics',
        'batch_size': len(candidates),
        'resources': resources,
        'normalized': normalize_data
    }

def benchmark_huggingface_evaluate(batch_size=2, normalize_data=True, prepared: Optional[PreparedData]=None):
    if not evaluate or not rouge or not bleu or not meteor:
        raise RuntimeError("HuggingFace Evaluate not available or models failed to load")
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    if prepared is None:
        candidates, references = get_test_data(batch_size, normalize=normalize_data)
    else:
        candidates, references = prepared.candidates, prepared.references
    
    start_time = time.perf_counter()
    
    # Parallel computation of metrics
    rouge_results = rouge.compute(predictions=candidates, references=references, use_stemmer=False)
    bleu_results = bleu.compute(predictions=candidates, references=references)
    meteor_results = meteor.compute(predictions=candidates, references=references)
    
    end_time = time.perf_counter()
    
    resources = monitor.get_resource_usage()
    
    return {
        'metrics': {
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL'],
            'bleu': bleu_results['bleu'],
            'meteor': meteor_results['meteor'],
            'wer': 'N/A',
            'token_f1': 'N/A',
            'jaccard': 'N/A',
        },
        'execution_time': end_time - start_time,
        'package': 'HuggingFace Evaluate',
        'batch_size': len(candidates),
        'resources': resources,
        'normalized': normalize_data
    }

def benchmark_sacrebleu(batch_size=2, normalize_data=True, prepared: Optional[PreparedData]=None):
    sacrebleu_mod = get_sacrebleu()
    if not sacrebleu_mod:
        raise RuntimeError("SacreBLEU not available")
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    if prepared is None:
        candidates, references = get_test_data(batch_size, normalize=normalize_data)
        refs_transposed = list(zip(*references))
    else:
        candidates, refs_transposed = prepared.candidates, prepared.refs_transposed
    
    start_time = time.perf_counter()
    bleu_score = sacrebleu_mod.corpus_bleu(candidates, refs_transposed)
    chrf_score = sacrebleu_mod.corpus_chrf(candidates, refs_transposed)
    end_time = time.perf_counter()
    
    resources = monitor.get_resource_usage()
    
    return {
        'metrics': {
            'bleu': bleu_score.score,
            'chrf': chrf_score.score,
            'rouge1': 'N/A',
            'rouge2': 'N/A',
            'rougeL': 'N/A',
            'meteor': 'N/A',
            'wer': 'N/A',
            'token_f1': 'N/A',
            'jaccard': 'N/A',
        },
        'execution_time': end_time - start_time,
        'package': 'SacreBLEU',
        'batch_size': len(candidates),
        'resources': resources,
        'normalized': normalize_data
    }

def benchmark_nltk_metrics(batch_size=2, normalize_data=True, prepared: Optional[PreparedData]=None):
    nltk_mods = get_nltk()
    if not nltk_mods:
        raise RuntimeError("NLTK not available")
    
    sentence_bleu = nltk_mods['sentence_bleu']
    meteor_score = nltk_mods['meteor_score']
    
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    if prepared is None or prepared.nltk_refs_tokens is None:
        candidates, references = get_test_data(batch_size, normalize=normalize_data)
        refs_tokens = [[ref.split() for ref in refs] for refs in references]
        cand_tokens = [cand.split() for cand in candidates]
    else:
        candidates = prepared.candidates
        refs_tokens = prepared.nltk_refs_tokens
        cand_tokens = prepared.nltk_cand_tokens
    
    start_time = time.perf_counter()
    
    # Vectorized computation
    bleu_scores = [sentence_bleu(ref_tokens_list, cand_tokens_single) 
                    for cand_tokens_single, ref_tokens_list in zip(cand_tokens, refs_tokens)]
    meteor_scores = [meteor_score(ref_tokens_list, cand_tokens_single)
                     for cand_tokens_single, ref_tokens_list in zip(cand_tokens, refs_tokens)]
    
    end_time = time.perf_counter()
    resources = monitor.get_resource_usage()
    
    return {
        'metrics': {
            'bleu': sum(bleu_scores) / len(bleu_scores),
            'meteor': sum(meteor_scores) / len(meteor_scores),
            'rouge1': 'N/A',
            'rouge2': 'N/A',
            'rougeL': 'N/A',
            'wer': 'N/A',
            'token_f1': 'N/A',
            'jaccard': 'N/A',
        },
        'execution_time': end_time - start_time,
        'package': 'NLTK',
        'batch_size': len(candidates),
        'resources': resources,
        'normalized': normalize_data
    }

def benchmark_rouge_score_metrics(batch_size=2, normalize_data=True, prepared: Optional[PreparedData]=None):
    if not HAS_ROUGE_SCORE:
        raise RuntimeError("rouge_score not available")
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    if prepared is None or prepared.rouge_scorer_obj is None:
        candidates, references = get_test_data(batch_size, normalize=normalize_data)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        refs_primary = [refs[0] for refs in references]
    else:
        candidates = prepared.candidates
        refs_primary = prepared.refs_primary
        scorer = prepared.rouge_scorer_obj
    
    start_time = time.perf_counter()
    
    # Vectorized scoring
    scores_list = [scorer.score(ref, candidate) for candidate, ref in zip(candidates, refs_primary)]
    
    r1_list = [scores['rouge1'].fmeasure for scores in scores_list]
    r2_list = [scores['rouge2'].fmeasure for scores in scores_list]
    rL_list = [scores['rougeL'].fmeasure for scores in scores_list]
    
    end_time = time.perf_counter()
    resources = monitor.get_resource_usage()
    
    return {
        'metrics': {
            'rouge1': sum(r1_list) / len(r1_list),
            'rouge2': sum(r2_list) / len(r2_list),
            'rougeL': sum(rL_list) / len(rL_list),
        },
        'execution_time': end_time - start_time,
        'package': 'rouge_score',
        'batch_size': len(candidates),
        'resources': resources,
        'normalized': normalize_data
    }

# Skip BertScore benchmark for speed unless explicitly enabled

# ==== Results Display Functions (simplified) ====

def print_compact_results(results: List[Dict], batch_size: int):
    """Compact results display for faster execution"""
    print(f"\n BENCHMARK RESULTS - BATCH SIZE: {batch_size}")
    print("-" * 60)
    
    for result in results:
        if 'error' in result:
            print(f" {result['package']}: {result['error']}")
            continue
            
        package = result['package']
        exec_time = result['execution_time']
        memory = result.get('resources', {}).get('memory_used_mb', 0)
        
        print(f" {package:20} | {exec_time:8.4f}s | {memory:6.1f} MB")

def print_speed_comparison(results: List[Dict]):
    """Fast speed comparison"""
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return
    
    blazemetrics_result = next((r for r in valid_results if r['package'] == 'BlazeMetrics'), None)
    if not blazemetrics_result:
        return
    
    blazemetrics_time = blazemetrics_result['execution_time']
    if blazemetrics_time == 0:
        return
        
    print(f"\n SPEED vs BlazeMetrics ({blazemetrics_time:.4f}s):")
    
    for result in valid_results:
        if result['package'] == 'BlazeMetrics':
            continue
            
        exec_time = result['execution_time']
        if exec_time > 0:
            ratio = exec_time / blazemetrics_time
            status = "🟢" if ratio < 1 else ""
            print(f"{status} {result['package']:20} | {ratio:5.2f}x")

# ==== Optimized Multi-run timing ====

def run_with_fewer_repeats(bench_fn, repeats: int, *, batch_size: int, normalize_data: bool, prepared: PreparedData):
    """Optimized timing with fewer repeats and early exit on consistent results"""
    times = []
    last_result = None
    
    for i in range(repeats):
        res = bench_fn(batch_size=batch_size, normalize_data=normalize_data, prepared=prepared)
        times.append(res['execution_time'])
        last_result = res
        
        # Early exit if we have consistent results (optional optimization)
        if i >= 2:  # At least 3 runs
            recent_times = times[-3:]
            if max(recent_times) - min(recent_times) < 0.01:  # Very consistent
                break
    
    times.sort()
    median_time = times[len(times)//2]
    last_result['execution_time'] = median_time
    return last_result

# ==== Optimized Public API ====

def run_text_benchmarks_fast(batch_sizes: List[int] = None, repeats: int = 3, normalize: bool = True) -> Dict[int, List[Dict]]:
    """Optimized benchmark runner with fewer repeats and streamlined execution"""
    if batch_sizes is None:
        batch_sizes = [1000]  # Reduced default size
    
    # Minimal warm-up
    minimal_warmup()
    
    all_results: Dict[int, List[Dict]] = {}
    
    for batch_size in batch_sizes:
        print(f"\n Running benchmarks for batch size: {batch_size}")
        
        # Pre-compute data once
        prepared = PreparedData(batch_size, normalize_data=normalize)
        
        # Select available benchmarks
        benchmarks = [benchmark_blazemetrics_text_metrics]
        
        # Check if evaluate and its models loaded successfully before adding the benchmark
        if evaluate and rouge and bleu and meteor:
            benchmarks.append(benchmark_huggingface_evaluate)
        else:
            print("   ️ Skipping HuggingFace Evaluate benchmark due to failed module/model loading.")
            
        if get_sacrebleu():
            benchmarks.append(benchmark_sacrebleu)
        if get_nltk():
            benchmarks.append(benchmark_nltk_metrics)
        if HAS_ROUGE_SCORE:
            benchmarks.append(benchmark_rouge_score_metrics)
        
        results: List[Dict] = []
        
        for bench in benchmarks:
            try:
                print(f"   ⏳ {bench.__name__.replace('benchmark_', '').replace('_', ' ').title()}...", end=' ')
                res = run_with_fewer_repeats(bench, repeats, batch_size=batch_size, normalize_data=normalize, prepared=prepared)
                results.append(res)
                print(f"{res['execution_time']:.4f}s")
            except Exception as e:
                error_result = {
                    'error': str(e)[:100], 
                    'package': bench.__name__.replace('benchmark_', '').replace('_', ' ').title(), 
                    'batch_size': batch_size
                }
                results.append(error_result)
                print(f" Failed")
        
        all_results[batch_size] = results
        print_compact_results(results, batch_size)
        print_speed_comparison(results)
    
    return all_results

# ==== Plotting Functions (NEW & UPDATED) ====

def prepare_data_for_plot(all_results: Dict[int, List[Dict]]):
    plot_data = []
    for batch_size, results in all_results.items():
        for res in results:
            if 'error' not in res:
                plot_data.append({
                    'package': res['package'],
                    'execution_time': res['execution_time'],
                    'memory_used_mb': res.get('resources', {}).get('memory_used_mb', 0),
                    'batch_size': batch_size
                })
    
    # Sort data by execution_time to make BlazeMetrics stand out and for better visualization
    plot_data.sort(key=lambda x: x['execution_time'])
    
    return plot_data

def create_and_save_plots(plot_data: List[Dict]):
    """Creates and saves a performance and a metrics plot using Plotly."""
    if not plot_data:
        print("No data available to plot.")
        return

    # Extract sorted data
    packages = [d['package'] for d in plot_data]
    execution_times = [d['execution_time'] for d in plot_data]
    memory_usages = [d['memory_used_mb'] for d in plot_data]

    # Assign colors: red for BlazeMetrics, light gray for others
    colors = ['#FF6347' if p == 'BlazeMetrics' else '#D3D3D3' for p in packages]

    # Create a figure with two subplots: one for time, one for memory
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Execution Time (seconds)", "Memory Usage (MB)"),
        horizontal_spacing=0.1
    )

    # Add bar chart for execution time
    fig.add_trace(
        go.Bar(
            x=packages,
            y=execution_times,
            name="Execution Time",
            marker_color=colors,
            text=[f'{t:.2f}s' for t in execution_times],
            textposition='auto',
            textfont=dict(color='black')
        ),
        row=1, col=1
    )

    # Add bar chart for memory usage
    fig.add_trace(
        go.Bar(
            x=packages,
            y=memory_usages,
            name="Memory Used",
            marker_color=colors,
            text=[f'{t:.2f}MB' for t in memory_usages],
            textposition='auto',
            textfont=dict(color='black')
        ),
        row=1, col=2
    )
    
    # Customize the layout for a beautiful and meaningful plot
    fig.update_layout(
        title_text="BlazeMetrics vs. Other Libraries Performance Benchmark",
        showlegend=False,
        # Improve font and background for a cleaner look
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes to sort and add titles
    fig.update_xaxes(
        categoryorder='total ascending', # Ensure packages are sorted on x-axis
        tickangle=-45,
        title_text="Package Name",
        showgrid=False
    )
    fig.update_yaxes(
        title_text="Time (seconds)",
        showgrid=True,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Memory (MB)",
        showgrid=True,
        row=1, col=2
    )
    
    # Add an annotation to highlight BlazeMetrics
    if 'BlazeMetrics' in packages:
        blaze_metrics_pos = packages.index('BlazeMetrics')
        fig.add_annotation(
            x=blaze_metrics_pos,
            y=execution_times[blaze_metrics_pos] + max(execution_times)*0.05,
            text="BlazeMetrics",
            showarrow=True,
            arrowhead=1,
            arrowcolor="#FF6347",
            font=dict(size=14, color="#FF6347", family="Arial Black")
        )
    
    # Save the plot to an HTML file and open it
    output_path = "blazemetrics_performance_comparison.html"
    fig.write_html(output_path, auto_open=True)
    print(f" Plot saved to {output_path}")

# ==== MAIN: Optimized Execution ====

if __name__ == "__main__":
    print(" FAST BlazeMetrics Benchmark Suite")
    print(" Optimized for speed while maintaining accuracy")
    
    # Reduced parameters for speed
    batch_sizes = [1000]  # Single batch size
    REPEATS = 3  # Fewer repeats
    NORMALIZE = True

    results = run_text_benchmarks_fast(batch_sizes, REPEATS, NORMALIZE)
    
    # 1. Prepare the data for plotting
    plot_data = prepare_data_for_plot(results)

    # 2. Create and save the plots
    create_and_save_plots(plot_data)