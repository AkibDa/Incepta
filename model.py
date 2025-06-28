import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, ShapEPipeline, DiffusionPipeline
from diffusers.utils import export_to_gif, export_to_video

# === Text-to-Image (Stable Diffusion) ===
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to("mps")

    image = pipe(prompt).images[0]
    image.save("image.png")

# === Text-to-Video (Wan-AI) ===
def generate_video(prompt):
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("mps")

    negative_prompt = "blurred, static, low quality"
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0
    ).frames[0]
    export_to_video(output, "output.mp4", fps=15)

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

    gif_path = export_to_gif(images, "3d.gif")

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