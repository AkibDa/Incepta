import os
import torch
from PIL import Image
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, ShapEPipeline, DiffusionPipeline
from diffusers.utils import export_to_gif, export_to_video
import tempfile

# === Text-to-Image (Stable Diffusion) ===
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to("mps")

    image = pipe(prompt).images[0]
    image.show()
    torch.mps.empty_cache()

# === Text-to-Video (Wan-AI) ===
def generate_video(prompt):
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("mps")

    negative_prompt = "blurred, static, low quality"
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=360,
        width=640,
        num_frames=45,
        guidance_scale=5.0
    ).frames[0]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name

    export_to_video(output, temp_path, fps=15)
    os.system(f'open "{temp_path}"')  # Use 'start' for Windows or 'xdg-open' for Linux
    torch.mps.empty_cache()

# === Text-to-3D (Shape-E) ===
def generate_3D(prompt):
    repo = "openai/shap-e"
    pipe = ShapEPipeline.from_pretrained(repo).to("mps")
    images = pipe(
        prompt,
        guidance_scale=15.0,
        num_inference_steps=64,
        size=256
    ).images
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_path = tmp.name

    export_to_gif(images, gif_path)
    Image.open(gif_path).show()
    torch.mps.empty_cache()

# === Prompt Enhancer (Text-to-Text) ===
def enhance_prompt(prompt):
    prompt_enhancer = pipeline("text2text-generation", model="google/flan-t5-large")

    instruction = "Make this prompt more vivid, detailed, and suitable for high-quality visual output."
    input_text = f"{instruction}: {prompt}"

    enhanced = prompt_enhancer(input_text, max_length=100)[0]['generated_text']
    print("Enhanced Prompt:", enhanced)
    return enhanced

def main():
    prompt = input("Enter prompt: ")
    enhanced = enhance_prompt(prompt)
    generate_image(enhanced)
    generate_video(enhanced)
    generate_3D(enhanced)

if __name__ == "__main__":
    main()
