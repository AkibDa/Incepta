import os
import torch
import gc
import tempfile
import streamlit as st
import requests
from transformers import pipeline
from diffusers import (
  StableDiffusionPipeline,
  EulerDiscreteScheduler,
  DiffusionPipeline,
  ShapEPipeline,
)
from diffusers.utils import export_to_gif, export_to_video
from io import BytesIO
from datetime import datetime

# === Constants and Config ===
MODEL_INFO = {
  "text_enhancer": "google/flan-t5-base",
  "image_generator": "CompVis/stable-diffusion-v1-4",
  "video_generator": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "3d_generator": "openai/shap-e"
}


# === Device Helpers ===
@st.cache_resource
def get_device_info():
  """Cache device detection to avoid repeated checks"""
  return {
    "cpu": "cpu",
    "mps": "mps" if torch.backends.mps.is_available() else "cpu",
    "cuda": 0 if torch.cuda.is_available() else -1
  }


def clear_memory():
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


# === Prompt History ===
def init_prompt_history():
  if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []


def add_to_history(prompt, enhanced_prompt, outputs, output_files):
  entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "original": prompt,
    "enhanced": enhanced_prompt,
    "outputs": outputs,
    "files": output_files
  }
  st.session_state.prompt_history.insert(0, entry)
  # Keep only last 5 entries to save memory
  if len(st.session_state.prompt_history) > 5:
    st.session_state.prompt_history = st.session_state.prompt_history[:5]


# === Enhanced Generation Functions ===
@st.cache_resource(show_spinner=False)
def load_prompt_enhancer():
  """Cache the enhancer model for performance"""
  return pipeline(
    "text2text-generation",
    model=MODEL_INFO["text_enhancer"],
    device=get_device_info()["cuda"]
  )

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def enhance_prompt(prompt, mode="Standard", creativity=0.7):
    with st.spinner("🧠 Enhancing your prompt..."):
        instructions = {
            "Standard": "Improve this prompt for general creative generation:",
            "Photorealistic": "Make this photorealistic with detailed descriptions:",
            "Artistic": "Enhance for artistic style with creative elements:",
            "Cinematic": "Optimize for cinematic visuals with dramatic elements:",
            "Detailed": "Add rich, specific details to this prompt:",
            "3D Render": "Prepare for 3D modeling with technical details:"
        }

        instruction = instructions.get(mode, "Improve:")
        input_text = f"{instruction} {prompt}"

        max_len = min(50 + int(creativity * 100), 150)

        payload = {
            "inputs": input_text,
            "parameters": {
                "max_length": max_len
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            try:
                enhanced = response.json()[0]["generated_text"]
            except Exception:
                st.error("Error parsing response.")
                return prompt
        else:
            st.error(f"API error: {response.status_code}")
            return prompt

        with st.expander("🔍 Prompt Enhancement Details", expanded=False):
            st.markdown(f"**Original:** `{prompt}`")
            st.markdown(f"**Enhanced:** `{enhanced}`")
            st.caption(f"Mode: {mode} | Creativity: {creativity:.1f}")

        return enhanced



@st.cache_resource(show_spinner=False)
def load_image_pipeline():
  """Cache the image generation pipeline"""
  device = get_device_info()["mps"]
  pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_INFO["image_generator"],
    scheduler=EulerDiscreteScheduler.from_pretrained(
      MODEL_INFO["image_generator"],
      subfolder="scheduler"
    ),
    torch_dtype=torch.float16 if device == "mps" else torch.float32
  ).to(device)
  return pipe


def generate_image(prompt, negative_prompt="", steps=20, guidance=7.5, seed=None):
  with st.spinner("🎨 Painting your vision..."):
    clear_memory()
    pipe = load_image_pipeline()
    device = get_device_info()["mps"]

    generator = torch.Generator(device).manual_seed(seed) if seed else None

    image = pipe(
      prompt,
      negative_prompt=negative_prompt,
      num_inference_steps=steps,
      guidance_scale=guidance,
      generator=generator
    ).images[0]

    # Convert to bytes for saving
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")

    clear_memory()
    return image, img_bytes.getvalue()


@st.cache_resource(show_spinner=False)
def load_video_pipeline():
  """Cache the video generation pipeline"""
  return DiffusionPipeline.from_pretrained(
    MODEL_INFO["video_generator"],
    torch_dtype=torch.float32
  ).to(get_device_info()["cpu"])


def generate_video(prompt, negative_prompt="", height=352, width=640, num_frames=33, guidance=5.0):
  with st.spinner("🎥 Directing your scene..."):
    clear_memory()
    pipe = load_video_pipeline()

    output = pipe(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=height,
      width=width,
      num_frames=num_frames,
      guidance_scale=guidance
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
      video_path = tmp.name
      export_to_video(output, video_path, fps=15)

    with open(video_path, "rb") as f:
      video_bytes = f.read()

    os.unlink(video_path)
    clear_memory()
    return video_path, video_bytes


@st.cache_resource(show_spinner=False)
def load_3d_pipeline():
  """Cache the 3D generation pipeline"""
  device = get_device_info()["mps"]
  pipe = ShapEPipeline.from_pretrained(MODEL_INFO["3d_generator"]).to(device)
  return pipe


def generate_3d(prompt, guidance=15.0, steps=64, size=256):
  with st.spinner("🧊 Sculpting your 3D model..."):
    clear_memory()
    pipe = load_3d_pipeline()

    images = pipe(
      prompt=prompt,
      guidance_scale=guidance,
      num_inference_steps=steps,
    ).images

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
      gif_path = tmp.name
      export_to_gif(images, gif_path)

    with open(gif_path, "rb") as f:
      gif_bytes = f.read()

    os.unlink(gif_path)
    clear_memory()
    return gif_path, gif_bytes


# === Streamlit UI ===
def main():
  # Page Config
  st.set_page_config(
    page_title="Incepta - AI Generation Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  # Initialize session state
  init_prompt_history()
  if 'advanced_expanded' not in st.session_state:
    st.session_state.advanced_expanded = False

  # Custom CSS
  st.markdown("""
    <style>
        .main {padding-top: 1.5rem;}
        .stTextArea textarea {min-height: 150px;}
        .footer {text-align: center; padding: 1rem; margin-top: 2rem;}
        .prompt-box {background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #e0e0e0;}
        .history-item {border-left: 3px solid #4CAF50; padding: 0.5rem 1rem; margin: 0.5rem 0; background-color: #f8f9fa; border-radius: 0.3rem;}
        .model-badge {background-color: #e9f7ef; padding: 0.2rem 0.5rem; border-radius: 0.5rem; font-size: 0.8rem; display: inline-block; margin: 0.2rem;}
        .status-box {padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; background-color: #f0f2f6;}
        .tab-content {padding: 1rem 0;}
    </style>
    """, unsafe_allow_html=True)

  # Header
  st.title("🚀 Incepta AI Generation Studio")
  st.markdown("""
    **Professional multi-modal generation pipeline**  
    *Enhance prompts and generate images, videos, and 3D models with fine-grained control*
    """)
  st.markdown("---")

  # Main Columns
  col1, col2 = st.columns([3, 2])

  with col1:
    # Prompt Input
    st.subheader("✏️ Your Creative Prompt")
    user_prompt = st.text_area(
      "Describe what you want to generate:",
      placeholder="A futuristic cityscape at sunset with flying cars and neon lights...",
      label_visibility="collapsed",
      help="Be as descriptive as possible for best results"
    )

    use_enhancer = st.checkbox(
      "Enable AI Prompt Enhancement",
      value=True,
      help="Automatically optimize your prompt for better generation results"
    )

  with col2:
    # Mode Selection
    st.subheader("🎛 Generation Mode")
    modes = {
      "🎨 Standard": "Balanced creativity and coherence",
      "📸 Photorealistic": "For realistic image generation",
      "🎭 Artistic": "Emphasizes stylistic interpretation",
      "📽 Cinematic": "Optimized for video generation",
      "🖌 Detailed": "Maximizes fine details",
      "🧊 3D Render": "Optimized for 3D model generation"
    }
    selected_mode = st.radio(
      "Select generation mode:",
      options=list(modes.keys()),
      format_func=lambda x: f"{x} - {modes[x]}",
      index=0,
      label_visibility="collapsed"
    )

  # Advanced Settings
  with st.sidebar:
    st.subheader("⚙️ Generation Controls")

    # Prompt Enhancement Controls
    if use_enhancer:
      creativity = st.slider(
        "Enhancement Creativity",
        min_value=0.1, max_value=1.0, value=0.7, step=0.1,
        help="How much the AI modifies your original prompt"
      )

    # Output Selection
    output_types = st.multiselect(
      "Output Types to Generate",
      ["Image", "Video", "3D Model"],
      default=["Image"],
      help="Select which media types to generate"
    )

    # Advanced Options
    with st.expander("🧪 Advanced Parameters", expanded=st.session_state.advanced_expanded):
      # Common params
      seed = st.number_input(
        "Random Seed",
        min_value=0, max_value=999999, value=42,
        help="For reproducible results"
      )

      # Image-specific
      if "Image" in output_types:
        st.markdown("**Image Settings**")
        img_steps = st.slider("Inference Steps", 10, 50, 20, key="img_steps")
        img_guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5, step=0.5, key="img_guidance")

      # Video-specific
      if "Video" in output_types:
        st.markdown("**Video Settings**")
        vid_frames = st.slider("Number of Frames", 10, 50, 33, key="vid_frames")
        vid_guidance = st.slider("Video Guidance", 1.0, 10.0, 5.0, step=0.5, key="vid_guidance")

      # 3D-specific
      if "3D Model" in output_types:
        st.markdown("**3D Settings**")
        three_d_guidance = st.slider("3D Guidance", 5.0, 20.0, 15.0, step=0.5, key="3d_guidance")
        three_d_steps = st.slider("3D Steps", 30, 100, 64, key="3d_steps")

      # Negative prompt
      use_negative = st.checkbox("Use Negative Prompt", True)
      negative_prompt = ""
      if use_negative:
        negative_prompt = st.text_area(
          "Negative Prompt",
          value="blurry, low quality, distorted, watermark",
          help="What to exclude from generation"
        )

  # Run Button
  st.markdown("---")
  if st.button("✨ Generate Content", type="primary", use_container_width=True):
    if not user_prompt:
      st.warning("Please enter a prompt first!")
      st.stop()

    # Enhance or use original prompt
    if use_enhancer:
      enhanced_prompt = enhance_prompt(user_prompt, selected_mode.split()[1], creativity)
    else:
      enhanced_prompt = user_prompt
      st.markdown(
        f'<div class="prompt-box">'
        f'<strong>Using Original Prompt:</strong><br>{user_prompt}'
        f'</div>',
        unsafe_allow_html=True
      )

    # Generate outputs
    outputs = {}
    output_files = {}

    with st.status("🚀 Generating content...", expanded=True) as status:
      if "Image" in output_types:
        status.update(label="🎨 Generating image...", state="running")
        try:
          img, img_bytes = generate_image(
            enhanced_prompt,
            negative_prompt,
            steps=img_steps,
            guidance=img_guidance,
            seed=seed
          )
          outputs["image"] = img
          output_files["image"] = img_bytes
          status.update(label="✅ Image generated!", state="complete")
        except Exception as e:
          status.update(label="❌ Image generation failed", state="error")
          st.error(f"Image generation error: {str(e)}")

      if "Video" in output_types:
        status.update(label="🎥 Generating video...", state="running")
        try:
          vid_path, vid_bytes = generate_video(
            enhanced_prompt,
            negative_prompt,
            num_frames=vid_frames,
            guidance=vid_guidance
          )
          outputs["video"] = vid_path
          output_files["video"] = vid_bytes
          status.update(label="✅ Video generated!", state="complete")
        except Exception as e:
          status.update(label="❌ Video generation failed", state="error")
          st.error(f"Video generation error: {str(e)}")

      if "3D Model" in output_types:
        status.update(label="🧊 Generating 3D model...", state="running")
        try:
          gif_path, gif_bytes = generate_3d(
            enhanced_prompt,
            guidance=three_d_guidance,
            steps=three_d_steps
          )
          outputs["3d_model"] = gif_path
          output_files["3d_model"] = gif_bytes
          status.update(label="✅ 3D model generated!", state="complete")
        except Exception as e:
          status.update(label="❌ 3D generation failed", state="error")
          st.error(f"3D generation error: {str(e)}")

    # Display results in tabs
    if outputs:
      st.markdown("---")
      st.subheader("🎉 Generation Results")

      tabs = st.tabs([f" {k} " for k in outputs.keys()])
      for tab, (out_type, out_val) in zip(tabs, outputs.items()):
        with tab:
          if out_type == "image":
            st.image(out_val, caption="Generated Image", use_column_width=True)
          elif out_type == "video":
            st.video(output_files[out_type])
          elif out_type == "3d_model":
            st.image(output_files[out_type], caption="3D Model Preview", use_column_width=True)

      # Download buttons
      st.markdown("---")
      st.subheader("💾 Download Options")
      cols = st.columns(3)

      if "image" in outputs:
        with cols[0]:
          st.download_button(
            "Download Image",
            data=output_files["image"],
            file_name="generated_image.png",
            mime="image/png",
            use_container_width=True
          )

      if "video" in outputs:
        with cols[1]:
          st.download_button(
            "Download Video",
            data=output_files["video"],
            file_name="generated_video.mp4",
            mime="video/mp4",
            use_container_width=True
          )

      if "3d_model" in outputs:
        with cols[2]:
          st.download_button(
            "Download 3D Preview",
            data=output_files["3d_model"],
            file_name="3d_preview.gif",
            mime="image/gif",
            use_container_width=True
          )

      # Add to history
      add_to_history(user_prompt, enhanced_prompt, list(outputs.keys()), output_files)

  # Prompt History Section
  if st.session_state.prompt_history:
    st.markdown("---")
    st.subheader("📜 Generation History")

    for i, entry in enumerate(st.session_state.prompt_history):
      with st.container():
        st.markdown(
          f'<div class="history-item">'
          f'<strong>{entry["timestamp"]}</strong> - {entry["original"][:50]}...<br>'
          f'<small>Generated: {", ".join(entry["outputs"])}</small>'
          f'</div>',
          unsafe_allow_html=True
        )

        cols = st.columns([4, 1])
        with cols[0]:
          with st.expander("View Details", expanded=False):
            st.markdown(f"**Original:** `{entry['original']}`")
            if entry["original"] != entry["enhanced"]:
              st.markdown(f"**Enhanced:** `{entry['enhanced']}`")

            # Show thumbnails if available
            if "image" in entry["files"]:
              st.image(entry["files"]["image"], caption="Generated Image", width=200)

        with cols[1]:
          if st.button(f"Reuse #{i + 1}", key=f"reuse_{i}"):
            st.session_state.reuse_prompt = entry["original"]
            st.rerun()

  # Tech Stack Footer
  st.markdown("---")
  st.markdown("### 🔧 Powered By")
  cols = st.columns(5)
  with cols[0]:
    st.markdown('<span class="model-badge" style="color:black">🤗 Transformers</span>', unsafe_allow_html=True)
  with cols[1]:
    st.markdown('<span class="model-badge" style="color:black">🖌️ Stable Diffusion</span>', unsafe_allow_html=True)
  with cols[2]:
    st.markdown('<span class="model-badge" style="color:black">🎥 Wan-AI</span>', unsafe_allow_html=True)
  with cols[3]:
    st.markdown('<span class="model-badge" style="color:black">🧊 Shap-E</span>', unsafe_allow_html=True)
  with cols[4]:
    st.markdown('<span class="model-badge" style="color:black">🚀 Streamlit</span>', unsafe_allow_html=True)

  st.markdown("""
    <div class="footer">
        <p style="color:white">Incepta v1.2 | © 2023 | <a href="https://github.com/AkibDa" target="_blank">GitHub</a></p>
        <small style="color:white">Note: Generation times may vary based on your hardware</small>
    </div>
    """, unsafe_allow_html=True)

  # Handle prompt reuse
  if 'reuse_prompt' in st.session_state:
    user_prompt = st.session_state.reuse_prompt
    del st.session_state.reuse_prompt
    st.rerun()


if __name__ == "__main__":
  main()