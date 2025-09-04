"""
Integration Example: Evidently with BlazeMetrics

Demonstrates how to use BlazeMetrics with Evidently for text drift and quality monitoring.
"""

from blazemetrics.integrations.evidently_integration import EvidentlyExporter

current_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a lazy dog."
]
reference_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog."
]

exporter = EvidentlyExporter()
report = exporter.generate_text_report(current_texts, reference_texts)

if report is not None:
    print("Evidently report generated successfully.")
else:
    print("Evidently report could not be generated (Evidently not installed?).")