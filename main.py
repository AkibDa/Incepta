import torch
import gc
import tempfile
import streamlit as st
from transformers import pipeline
from diffusers import (
  StableDiffusionPipeline,
  EulerDiscreteScheduler,
  DiffusionPipeline,
  ShapEPipeline,
)
from diffusers.utils import export_to_gif, export_to_video
from streamlit.components.v1 import html
from io import BytesIO


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
def enhance_prompt(prompt, mode="Standard"):
  with st.spinner("🧠 Enhancing your prompt..."):
    instruction = {
      "Standard": "Improve this prompt for general creative generation:",
      "Photorealistic": "Make this prompt more suitable for photorealistic image generation:",
      "Artistic": "Enhance this prompt for artistic style generation:",
      "Cinematic": "Optimize this prompt for cinematic video generation:",
      "Detailed": "Add rich details to this prompt for high-resolution output:",
      "3D Render": "Prepare this prompt for 3D model generation:"
    }.get(mode, "Improve this prompt:")

    prompt_enhancer = pipeline(
      "text2text-generation",
      model="google/flan-t5-large",
      device=0 if torch.cuda.is_available() else -1
    )
    input_text = f"{instruction} {prompt}"
    enhanced = prompt_enhancer(input_text, max_length=100)[0]['generated_text']

    # Display the enhanced prompt in a styled box
    st.markdown(
      f'<div style="padding:15px; border-radius:5px; margin-top:10px; border:1px solid #ddd">'
      f'<strong>✨ Enhanced Prompt:</strong><br>{enhanced}'
      f'</div>',
      unsafe_allow_html=True
    )

    return enhanced


# === Generation Functions ===
def generate_image(prompt, negative_prompt=""):
  with st.spinner("🎨 Generating image..."):
    clear_memory()
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
      model_id,
      scheduler=scheduler,
      torch_dtype=torch.float16 if get_mps() == "mps" else torch.float32
    ).to(get_mps())

    image = pipe(
      prompt,
      negative_prompt=negative_prompt,
      num_inference_steps=20
    ).images[0]

    clear_memory()
    return image


def generate_video(prompt, negative_prompt=""):
  with st.spinner("🎥 Generating video..."):
    clear_memory()
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    pipe = DiffusionPipeline.from_pretrained(
      model_id,
      torch_dtype=torch.float32
    ).to(get_cpu())

    output = pipe(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=352,
      width=640,
      num_frames=33,
      guidance_scale=5.0
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
      video_path = tmp.name

    export_to_video(output, video_path, fps=15)
    clear_memory()
    return video_path


def generate_3D(prompt):
  with st.spinner("🧊 Generating 3D model..."):
    clear_memory()
    repo = "openai/shap-e"
    device = get_mps()

    pipe = ShapEPipeline.from_pretrained(repo).to(device)

    images = pipe(
      prompt=prompt,
      guidance_scale=15.0,
      num_inference_steps=64,
    ).images

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
      gif_path = tmp.name

    export_to_gif(images, gif_path)
    clear_memory()
    return gif_path


# === Streamlit UI ===
def main():
  st.set_page_config(
    page_title="Incepta - AI Generation Studio",
    page_icon="🚀",
    layout="wide"
  )

  # Custom CSS
  st.markdown("""
    <style>
        .main {padding-top: 2rem;}
        .stTextArea textarea {min-height: 150px;}
        .footer {position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px;}
        .enhanced-prompt {background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-top: 10px; border: 1px solid #ddd;}
        .mode-card {border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px;}
        .selected-mode {border: 2px solid #4CAF50;}
        .generation-tab {padding: 15px; border-radius: 10px; margin-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

  # Session state
  if 'enhanced_prompt' not in st.session_state:
    st.session_state.enhanced_prompt = ""
  if 'generated_outputs' not in st.session_state:
    st.session_state.generated_outputs = {}

  # Header
  st.title("🚀 Incepta AI Generation Studio")
  st.markdown("""
    **Multi-Modal Generation Pipeline**  
    *Enhance prompts and generate images, videos, and 3D models in one workflow.*
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
      label_visibility="collapsed"
    )

    use_enhancer = st.checkbox(
      "Enable AI Prompt Enhancement",
      value=True,
      help="Optimize your prompt for better generation results"
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
    st.subheader("⚙️ Advanced Settings")
    use_negative_prompt = st.checkbox("Use negative prompt", value=True)
    negative_prompt = ""
    if use_negative_prompt:
      negative_prompt = st.text_input(
        "Negative prompt (what to exclude):",
        value="blurry, low quality, distorted, watermark"
      )

    st.markdown("---")
    st.caption("Generation Options:")
    output_types = st.multiselect(
      "Select output types to generate:",
      ["Image", "Video", "3D Model"],
      default=["Image"]
    )

  # Run Button
  st.markdown("---")
  if st.button("✨ Generate Content", type="primary", use_container_width=True):
    if not user_prompt:
      st.warning("Please enter a prompt first!")
      return

    # Enhance or use original prompt
    if use_enhancer:
      st.session_state.enhanced_prompt = enhance_prompt(user_prompt, selected_mode.split()[1])
    else:
      st.session_state.enhanced_prompt = user_prompt
      st.markdown(
        f'<div style="background-color:#f0f8ff; padding:15px; border-radius:5px; margin-top:10px; border:1px solid #ddd">'
        f'<strong>Using Original Prompt:</strong><br>{user_prompt}'
        f'</div>',
        unsafe_allow_html=True
      )

    # Generate outputs
    st.session_state.generated_outputs = {}
    prompt_to_use = st.session_state.enhanced_prompt

    # Display outputs in tabs
    tab1, tab2, tab3 = st.tabs(["Image", "Video", "3D Model"])

    if "Image" in output_types:
      with tab1:
        try:
          image = generate_image(prompt_to_use, negative_prompt)
          st.image(image, caption="Generated Image", use_column_width=True)
          st.session_state.generated_outputs["image"] = image
        except Exception as e:
          st.error(f"Failed to generate image: {str(e)}")

    if "Video" in output_types:
      with tab2:
        try:
          video_path = generate_video(prompt_to_use, negative_prompt)
          st.video(video_path)
          st.session_state.generated_outputs["video"] = video_path
        except Exception as e:
          st.error(f"Failed to generate video: {str(e)}")

    if "3D Model" in output_types:
      with tab3:
        try:
          gif_path = generate_3D(prompt_to_use)
          st.image(gif_path, caption="3D Model Preview", use_column_width=True)
          st.session_state.generated_outputs["3d_model"] = gif_path
        except Exception as e:
          st.error(f"Failed to generate 3D model: {str(e)}")

    # Download buttons
    if st.session_state.generated_outputs:
      st.markdown("---")
      st.subheader("Download Options")
      cols = st.columns(3)

      if "image" in st.session_state.generated_outputs:
        with cols[0]:
          buf = BytesIO()
          st.session_state.generated_outputs["image"].save(buf, format="PNG")
          st.download_button(
            "Download Image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
          )

      if "video" in st.session_state.generated_outputs:
        with cols[1]:
          with open(st.session_state.generated_outputs["video"], "rb") as f:
            st.download_button(
              "Download Video",
              data=f,
              file_name="generated_video.mp4",
              mime="video/mp4"
            )

      if "3d_model" in st.session_state.generated_outputs:
        with cols[2]:
          with open(st.session_state.generated_outputs["3d_model"], "rb") as f:
            st.download_button(
              "Download 3D Preview",
              data=f.read(),
              file_name="3d_preview.gif",
              mime="image/gif"
            )

  # Footer
  st.markdown("---")
  footer = """
    <div class="footer">
        <p style="color:white">Incepta [From Imagination to Reality — One Prompt at a Time] v1.0 | <a href="https://github.com/AkibDa" target="_blank">GitHub</a> | Made with ❤️ using Diffusers</p>
    </div>
    """
  html(footer, height=50)


if __name__ == "__main__":
  main()