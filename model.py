import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, ShapEPipeline, DiffusionPipeline
from diffusers.utils import export_to_gif, export_to_video

# === Prompt Enhancer (Text-to-Text) ===
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", torch_dtype=torch.float16, device_map="auto")

inputs = tokenizer("Enhance this prompt for an image generation: A cat", return_tensors="pt")
inputs = {k: v.to("mps") for k, v in inputs.items()}
output = model.generate(**inputs)
enhanced_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
print("Enhanced Prompt:", enhanced_prompt)

# === Text-to-Image (Stable Diffusion) ===
model_id = "CompVis/stable-diffusion-v1-4"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to("mps")

image = pipe(enhanced_prompt).images[0]
image.save("astronaut_rides_horse.png")

# === Text-to-Video (Wan-AI) ===
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("mps")

video_prompt = "A cat walks on the grass, realistic"
negative_prompt = "blurred, static, low quality"
output = pipe(
    prompt=video_prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=15)

# === Text-to-3D (Shape-E) ===
repo = "openai/shap-e"
pipe = ShapEPipeline.from_pretrained(repo).to("mps")
shape_prompt = "a shark"
images = pipe(
    shape_prompt,
    guidance_scale=15.0,
    num_inference_steps=64,
    size=256
).images

gif_path = export_to_gif(images, "shark_3d.gif")
