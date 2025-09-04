"""
Example: Multimodal AI Evaluation Suite

This script demonstrates how to use the MultimodalEvaluator from blazemetrics
to evaluate cross-modal models (text+image) and text-to-image generation.
"""

from blazemetrics import MultimodalEvaluator

# Example vision-language evaluation
questions = [
    "What is shown in the image?",
    "Describe the scenery in the photo."
]
image_paths = [
    "image1.png",
    "image2.png"
]
model_responses = [
    "A cat sitting on the sofa.",
    "A beautiful mountain landscape."
]

evaluator = MultimodalEvaluator()

results = evaluator.evaluate(
    inputs={"text": questions, "images": image_paths},
    outputs=model_responses,
    modalities=['text', 'vision'],
    metrics=['cross_modal_alignment', 'visual_grounding', 'multimodal_hallucination']
)
print("Vision-Language Evaluation:", results)

# Example text-to-image generation eval
text_prompts = [
    "A painting of a sunset over a lake",
    "A futuristic city skyline at night"
]
output_images = [
    "gen_image1.png",
    "gen_image2.png"
]
ground_truth_images = [
    "gt_image1.png",
    "gt_image2.png"
]

gen_results = evaluator.evaluate_generation(
    prompts=text_prompts,
    generated_images=output_images,
    reference_images=ground_truth_images,
    metrics=['clip_score', 'fid', 'inception_score', 'semantic_alignment']
)
print("Text-to-Image Evaluation:", gen_results)