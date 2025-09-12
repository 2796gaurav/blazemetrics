"""
13_multimodal_eval.py

BlazeMetrics Example – Multimodal Evaluation (Text+Image)
------------------------------------------------------
Illustrates evaluation of outputs involving more than one modality—e.g., text-to-image, vision+text captioning.
Key features demonstrated:
  - Cross-modal alignment (how well does generated text match the image, or vice versa?)
  - Generation metrics for multimodal outputs (CLIP score, visual grounding, etc. as supported)
  - Extendable to other modalities such as audio/video

Usage Recommendation:
- Good foundation for anyone building or benchmarking multimodal or text-to-image models.
- Shows both text/image alignment and full generation evaluation
"""
from blazemetrics import MultimodalEvaluator

eval = MultimodalEvaluator()

inputs = {
    "text": [
        "A cat sitting on a mat.",
        "The sun rises over mountains."
    ],
    "images": [
        "cat.jpg",
        "sunrise.png"
    ]
}
outputs = [
    "A black cat sits on a rug.",
    "Sun rises behind tall mountains."
]
modalities = ["text", "vision"]
result = eval.evaluate(inputs, outputs, modalities)
print("--- Multimodal alignment scores ---")
for k, v in result.items():
    print(f"  {k}: {v:.3f}")

# Generation evaluation for text-to-image
prompts = ["A dog running."]
generated_images = ["dog1.png"]
reference_images = ["dog_ref.png"]
gen_result = eval.evaluate_generation(prompts, generated_images, reference_images)
print("\n--- Generation metrics ---")
for k, v in gen_result.items():
    print(f"  {k}: {v:.3f}")
