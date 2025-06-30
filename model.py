import os
import torch
import gc
import tempfile
from PIL import Image
import requests
from transformers import pipeline
from diffusers import (
  StableDiffusionPipeline,
  EulerDiscreteScheduler,
  DiffusionPipeline,
  ShapEPipeline,
)
from diffusers.utils import export_to_gif, export_to_video


# === Device Helpers ===
def get_cpu():
  return "cpu"


def get_mps():
  return "mps" if torch.backends.mps.is_available() else "cpu"


def clear_memory():
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# === Prompt Enhancer (Text-to-Text) ===
def enhance_prompt(prompt):
  HF_TOKEN = os.environ.get("HF_TOKEN")

  API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
  headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
  }
  st.write("⏳ Enhancing prompt via Hugging Face Inference API...")

  instruction = "Make this prompt more vivid, detailed, and suitable for high-quality visual output."
  input_text = f"{instruction}: {prompt}"

  payload = {"inputs": input_text, "parameters": {"max_length": 100}}

  response = requests.post(API_URL, headers=headers, json=payload)

  if response.status_code == 200:
    try:
      enhanced = response.json()[0]["generated_text"]
      st.success("✨ Enhanced Prompt Generated")
      return enhanced
    except Exception as e:
      st.error("Failed to parse response.")
      return ""
  else:
    st.error(f"API Error: {response.status_code}")
    return ""


# === Text-to-Image (Stable Diffusion) ===
def generate_image(prompt):
  print("\n🖼 Generating image...")
  clear_memory()
  model_id = "stabilityai/stable-diffusion-2-1"
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16 if get_mps() == "mps" else torch.float32
  ).to(get_mps())

  image = pipe(prompt, num_inference_steps=20).images[0]  # Reduce steps to avoid crash
  image.show()
  clear_memory()


# === Text-to-Video (Wan-AI) ===
def generate_video(prompt):
  print("\n🎥 Generating video...")
  clear_memory()
  model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32  # CPU only
  ).to(get_cpu())

  output = pipe(
    prompt=prompt,
    negative_prompt="blurred, static, low quality",
    height=352,
    width=640,
    num_frames=33,
    guidance_scale=5.0
  ).frames[0]

  with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
    video_path = tmp.name

  export_to_video(output, video_path, fps=15)
  os.system(f'open "{video_path}"')
  clear_memory()


# === Text-to-3D (Shape-E) ===
def generate_3D(prompt):
  print("\n🧊 Generating 3D (Shape-E)...")
  clear_memory()
  repo = "openai/shap-e"
  device = get_mps()

  pipe = ShapEPipeline.from_pretrained(repo).to(device)

  images = pipe(
    prompt=prompt,
    guidance_scale=15.0,
    num_inference_steps=64,
    size=256
  ).images

  with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
    gif_path = tmp.name

  export_to_gif(images, gif_path)
  Image.open(gif_path).show()
  clear_memory()


# === Main Controller ===
def main():
  prompt = input("Enter prompt: ")
  enhanced = enhance_prompt(prompt)

  generate_image(enhanced)
  generate_video(enhanced)
  generate_3D(enhanced)


if __name__ == "__main__":
  main()
