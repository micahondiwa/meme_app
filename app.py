import diffusers
import streamlit as st
import torch
from PIL import ImageDraw, ImageFont
from diffusers import AutoPipelineForText2Image
from PIL import ImageDraw, ImageFont
import logging
from peft import PeftModel


LORA_WEIGHTS = "onstage3890/maya_model_v1_lora"

# Configure device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

def load_model():
    try:
        # Check if torch is properly installed
        if not hasattr(torch, 'float16'):
            raise ImportError("PyTorch is not properly installed.")
        
        # Check if PEFT is available
        try:
            import peft
        except ImportError:
            raise ImportError("PEFT library is required for LoRA weights. Install with: pip install peft")
        
        # Load the base model
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            safety_checker=None
        )
        
        # Load LoRA weights with PEFT backend
        pipeline.load_lora_weights(
            LORA_WEIGHTS,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="maya_adapter"
        )
        
        # Optimize pipeline
        pipeline.to(device)
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
            pipeline.enable_xformers_memory_efficient_attention()
        
        return pipeline
        
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

    
def generate_images(prompt, pipeline, n):
    return pipeline([prompt] * n).images

def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    # Initialization
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Calculate the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position at which to draw the text to center it
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)

def generate_memes(prompt, text, pipeline, n):
    images = generate_images(prompt, pipeline, n)
    for img in images:
        add_text_to_image(img, text) 
    return images

def main():
    st.title("Diffusion Model Image Generator")

    st.number_input("Enter a value", min_value=0, max_value=100, value=10, step=1)
if __name__ == "__main__":
    main()