"""
14_multimodal_video_audio_eval.py

BlazeMetrics Example – Multimodal Evaluation (Text, Audio, Video)
----------------------------------------------------------------
Illustrates advanced use for evaluating text-to-audio, text-to-video models:
  - Hallucination/alignment detection between text and audio/video outputs
  - Metrics: cross_modal_alignment, visual_grounding, others as supported by package
  - Demo covers both audio and video evaluation paths

Use this as reference for:
- Evaluating any system that goes beyond simple text/image (true multimodal pipelines)
- Auditing hallucination or grounding in generated media
"""
from blazemetrics import MultimodalEvaluator

eval = MultimodalEvaluator()

# Text/audio example
audio_prompts = ["Dog barking.", "Music playing."]
audio_outputs = ["Dog barking sound.", "Random noise."]
audio_modalities = ["text", "audio"]

audio_result = eval.evaluate({"text": audio_prompts, "audio": audio_outputs}, audio_outputs, audio_modalities, metrics=["cross_modal_alignment"])
print("-- Text <-> Audio Cross-Modal Alignment --")
for k, v in audio_result.items():
    print(f"  {k}: {v}")

# Text/video example
video_prompts = ["A cat jumps over a table.", "Sunny beach."]
video_outputs = ["video1.mp4", "video2.mp4"]
video_modalities = ["text", "video"]
video_result = eval.evaluate({"text": video_prompts, "video": video_outputs}, video_outputs, video_modalities, metrics=["cross_modal_alignment", "visual_grounding"])
print("-- Text <-> Video Alignment --")
for k, v in video_result.items():
    print(f"  {k}: {v}")
