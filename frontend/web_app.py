"""
Gradio Frontend for Stable Diffusion XL with LoRA
Provides an interactive web interface for image generation
"""

import os
import base64
import io
import logging
from typing import Optional

import gradio as gr
import requests
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    # IMPORTANT: Replace with your actual ngrok URL from the backend
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_NUM_STEPS = 50
    TIMEOUT = 120  # seconds

def decode_base64_image(image_b64: str) -> Image.Image:
    """
    Decode base64-encoded image string to PIL Image
    
    Args:
        image_b64: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    try:
        image_data = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        raise ValueError(f"Failed to decode image: {str(e)}")

def generate_image(
    prompt: str,
    guidance_scale: float,
    num_steps: int,
    progress=gr.Progress()
) -> tuple[Optional[Image.Image], str]:
    """
    Generate image from text prompt by calling the backend API
    
    Args:
        prompt: Text description of the desired image
        guidance_scale: How closely to follow the prompt (1.0-20.0)
        num_steps: Number of denoising steps (10-100)
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (generated_image, status_message)
    """
    try:
        # Validate inputs
        if not prompt or not prompt.strip():
            return None, "❌ Please enter a prompt"
        
        if len(prompt) > 500:
            return None, "❌ Prompt too long (max 500 characters)"
        
        logger.info(f"Generating image for prompt: '{prompt[:50]}...'")
        
        # Show progress
        progress(0, desc="Sending request to backend...")
        
        # Prepare request
        url = f"{Config.BACKEND_URL}/generate"
        payload = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps
        }
        
        # Make request
        progress(0.3, desc="Generating image...")
        logger.info(f"Sending request to {url}")
        
        response = requests.post(
            url,
            json=payload,
            timeout=Config.TIMEOUT
        )
        
        # Check response
        if response.status_code != 200:
            error_msg = response.json().get('error', 'Unknown error')
            logger.error(f"Backend error: {error_msg}")
            return None, f"❌ Error: {error_msg}"
        
        # Decode image
        progress(0.9, desc="Processing image...")
        data = response.json()
        image_b64 = data.get('image')
        
        if not image_b64:
            return None, "❌ No image received from backend"
        
        image = decode_base64_image(image_b64)
        
        logger.info("Image generated successfully")
        progress(1.0, desc="Complete!")
        
        return image, f"✅ Generated successfully! ({num_steps} steps, guidance {guidance_scale})"
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return None, "❌ Request timeout. Try reducing the number of steps."
    
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return None, f"❌ Cannot connect to backend at {Config.BACKEND_URL}. Is the backend running?"
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None, f"❌ Error: {str(e)}"

def check_backend_health() -> str:
    """
    Check if backend is accessible
    
    Returns:
        Status message
    """
    try:
        response = requests.get(
            f"{Config.BACKEND_URL}/health",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            device = data.get('device', 'unknown')
            return f"✅ Backend is healthy (running on {device})"
        else:
            return "⚠️ Backend returned unexpected status"
            
    except Exception as e:
        return f"❌ Backend not accessible: {str(e)}"

# Example prompts for quick testing
EXAMPLE_PROMPTS = [
    "A golden retriever playing in the snow, photorealistic, high quality",
    "A sunset over a mountain range, vibrant colors, dramatic lighting",
    "A cozy cabin in a snowy forest, winter wonderland, detailed",
    "An astronaut riding a horse on Mars, cinematic, 8k resolution",
    "A futuristic cityscape at night, neon lights, cyberpunk style"
]

def create_interface():
    """
    Create and configure the Gradio interface
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(
        title="Stable Diffusion XL + LoRA",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            """
            # 🎨 Stable Diffusion XL with LoRA
            
            Generate high-quality images from text descriptions using our fine-tuned SDXL model.
            
            **Tips for better results:**
            - Be specific and descriptive in your prompts
            - Include style keywords (e.g., "photorealistic", "oil painting", "anime")
            - Mention quality terms (e.g., "high quality", "detailed", "8k")
            - Adjust guidance scale: higher = closer to prompt, lower = more creative
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    max_lines=5
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=Config.DEFAULT_GUIDANCE_SCALE,
                        step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt (higher = more literal)"
                    )
                    
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=Config.DEFAULT_NUM_STEPS,
                        step=5,
                        label="Inference Steps",
                        info="More steps = better quality but slower"
                    )
                
                generate_btn = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    size="lg"
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                # Example prompts
                gr.Markdown("### 💡 Example Prompts")
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_PROMPTS],
                    inputs=[prompt_input],
                    label=None
                )
                
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=512
                )
        
        # Backend status
        with gr.Row():
            backend_status = gr.Textbox(
                label="Backend Status",
                interactive=False,
                value=check_backend_health()
            )
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, guidance_scale, num_steps],
            outputs=[output_image, status_text]
        )
        
        refresh_btn.click(
            fn=check_backend_health,
            outputs=[backend_status]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Note:** Image generation may take 30-60 seconds depending on the number of steps.
            Make sure the backend is running before generating images.
            """
        )
    
    return demo

def main():
    """Main entry point"""
    try:
        logger.info("Starting Gradio interface...")
        
        # Check backend connection
        logger.info(f"Backend URL: {Config.BACKEND_URL}")
        health_status = check_backend_health()
        logger.info(f"Backend status: {health_status}")
        
        # Create and launch interface
        demo = create_interface()
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Set to True for public sharing
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start interface: {str(e)}")
        raise

if __name__ == '__main__':
    main()