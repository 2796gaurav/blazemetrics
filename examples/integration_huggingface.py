"""
Example: HuggingFace Integration with BlazeMetrics

This script demonstrates how to use BlazeMetrics with HuggingFace pipelines.
"""

from blazemetrics.integrations.huggingface import with_blazemetrics_pipeline
from blazemetrics.client import BlazeMetricsClient

from transformers import pipeline

# Create a HuggingFace text generation pipeline
hf_pipeline = pipeline("text-generation", model="gpt2")

# Wrap the pipeline with BlazeMetrics for evaluation
client = BlazeMetricsClient()
blaze_pipeline = with_blazemetrics_pipeline(
    task="text-generation",
    pipeline_factory=pipeline,
    client=client,
    model="gpt2"
)

# Generate and evaluate
results = blaze_pipeline("The quick brown fox", references=[["The quick brown fox jumps over the lazy dog."]])
print("BlazeMetrics + HuggingFace Results:", results)