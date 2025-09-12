"""
09_code_security_eval.py

BlazeMetrics Example – Code Generation, Security, Performance, and Style Evaluation
----------------------------------------------------------------------------------
Overview:
    - Automatically rates generated code samples for:
        * Correctness (matches ground truth/reference)
        * Security (flags dangerous patterns, unsafe API use, injection, etc.)
        * Performance and efficiency (detects e.g. slow or naive algorithms)
        * Style and formatting (PEP8, naming conventions, indentation, etc.)
    - Supports language-specific checks (Python, JS, Java, and more)
    - Flexible batch API with comprehensive results for each prompt/sample

Parameters:
    - languages: List of programming languages, one for each code prompt
    - security_checks: Enable/disable automated security vulnerability checks
    - performance_analysis: Enable/disable performance/efficiency evaluation
    - metrics: (optional) List of metrics to report ("correctness", "security", "performance", "style", ...)

Returns:
    - A dict keyed by code sample, each with:
        * correctness_score: How well output matches expected/reference solution(s)
        * security_issues: List of flagged risky patterns or an empty list
        * performance: Textual judgment ("efficient", "inefficient", etc.)
        * style_compliance: Pass/fail with optional details
        * summary: Human-readable summary of issues/strengths

Usage:
    1. Prepare lists of prompts/tasks, generated code samples, references, and languages.
    2. Construct CodeEvaluator and call `.evaluate()`.
    3. Review the per-sample result dict, highlighting flagged issues.

See the docstring of blazemetrics.blazemetrics.CodeEvaluator for more details.

Sample:
"""
from blazemetrics import CodeEvaluator

prompts = [
    "Write a Python function to delete a file.",
    "Calculate Fibonacci numbers using recursion in Python.",
    "Open a socket and listen for connections."
]
generated_code = [
    "import os\ndef rmfile(path):\n  os.remove(path)",
    "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
    "import socket\ns = socket.socket(); s.bind(('', 8080)); s.listen(1)"
]
reference_solutions = [
    "import os\ndef remove_file(path):\n  os.unlink(path)",
    "def fib(num):\n  if num < 2: return num\n  return fib(num-1) + fib(num-2)",
    "import socket\ns = socket.socket(); s.bind(('localhost', 8888)); s.listen(5)"
]
languages = ["python", "python", "python"]

# You may also explicitly control metrics:
metrics = ["correctness", "security", "performance", "style"]
code_eval = CodeEvaluator(languages=languages, security_checks=True, performance_analysis=True)
results = code_eval.evaluate(prompts, generated_code, reference_solutions, metrics=metrics)

code_eval = CodeEvaluator(
    languages=languages,
    security_checks=True,         # Set False to skip security scan
    performance_analysis=True     # Set False to skip efficiency checks
)
results = code_eval.evaluate(prompts, generated_code, reference_solutions)
print("--- Code Security/Style Evaluation ---")
for k, v in results.items():
    print(f"  {k}: {v}")

"""
OUTPUT EXAMPLE:

--- Code Security/Style Evaluation ---
  0: {
        'correctness_score': 1.0,
        'security_issues': [],
        'performance': 'efficient',
        'style_compliance': 'pass',
        'summary': "No issues detected"
     }
  1: {
        'correctness_score': 0.9,
        'security_issues': [],
        'performance': 'inefficient: recursion used',
        'style_compliance': 'pass',
        'summary': "Performance warning: naive recursion"
     }
  2: {
        'correctness_score': 0.8,
        'security_issues': ['Bind to all interfaces', 'Low listen backlog'],
        'performance': 'efficient',
        'style_compliance': 'fail: non-PEP8 naming',
        'summary': "Security warning: binding socket to all interfaces"
     }
"""