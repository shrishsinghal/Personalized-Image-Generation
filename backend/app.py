"""
Flask Backend for Stable Diffusion XL with LoRA
Provides REST API endpoints for image generation
"""

import os
import base64
import io
import logging
from typing import Optional

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import DiffusionPipeline, AutoencoderKL
from pyngrok import ngrok

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
class Config:
    """Application configuration"""
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', '2qHIXBs799JYdxrdkFTAmvFuPkv_5WX6YVaoReGpiCwSFTeza')
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_WEIGHTS = "nikhilsoni700/dl_project_LoRA"
    VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_NUM_INFERENCE_STEPS = 50
    MAX_INFERENCE_STEPS = 100
    MIN_INFERENCE_STEPS = 10

# Global pipeline instance
pipeline: Optional[DiffusionPipeline] = None

def initialize_pipeline():
    """
    Initialize the Stable Diffusion XL pipeline with LoRA weights
    
    Returns:
        DiffusionPipeline: Initialized pipeline ready for inference
    """
    global pipeline
    
    try:
        logger.info("Initializing Stable Diffusion XL pipeline...")
        
        # Check device availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Running on CPU (will be slow).")
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Load VAE
        logger.info(f"Loading VAE from {Config.VAE_ID}")
        vae = AutoencoderKL.from_pretrained(
            Config.VAE_ID,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32
        )
        
        # Load base model
        logger.info(f"Loading base model from {Config.MODEL_ID}")
        pipeline = DiffusionPipeline.from_pretrained(
            Config.MODEL_ID,
            vae=vae,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            variant="fp16" if Config.DEVICE == "cuda" else None,
            use_safetensors=True
        )
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {Config.LORA_WEIGHTS}")
        pipeline.load_lora_weights(Config.LORA_WEIGHTS)
        
        # Move to device
        pipeline = pipeline.to(Config.DEVICE)
        
        logger.info("Pipeline initialized successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise

def validate_request_data(data: dict) -> tuple[bool, Optional[str]]:
    """
    Validate incoming request data
    
    Args:
        data: Request JSON data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"
    
    if 'prompt' not in data or not data['prompt']:
        return False, "Prompt is required"
    
    if len(data['prompt']) > 500:
        return False, "Prompt too long (max 500 characters)"
    
    # Validate guidance_scale
    if 'guidance_scale' in data:
        try:
            scale = float(data['guidance_scale'])
            if scale < 1.0 or scale > 20.0:
                return False, "guidance_scale must be between 1.0 and 20.0"
        except ValueError:
            return False, "guidance_scale must be a number"
    
    # Validate num_inference_steps
    if 'num_inference_steps' in data:
        try:
            steps = int(data['num_inference_steps'])
            if steps < Config.MIN_INFERENCE_STEPS or steps > Config.MAX_INFERENCE_STEPS:
                return False, f"num_inference_steps must be between {Config.MIN_INFERENCE_STEPS} and {Config.MAX_INFERENCE_STEPS}"
        except ValueError:
            return False, "num_inference_steps must be an integer"
    
    return True, None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': Config.DEVICE,
        'model_loaded': pipeline is not None
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    Generate image from text prompt
    
    Expected JSON:
    {
        "prompt": "text description",
        "guidance_scale": 7.5 (optional),
        "num_inference_steps": 50 (optional)
    }
    
    Returns:
        JSON with base64 encoded image
    """
    try:
        # Validate request
        data = request.get_json()
        is_valid, error_msg = validate_request_data(data)
        
        if not is_valid:
            logger.warning(f"Invalid request: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Extract parameters
        prompt = data['prompt']
        guidance_scale = float(data.get('guidance_scale', Config.DEFAULT_GUIDANCE_SCALE))
        num_inference_steps = int(data.get('num_inference_steps', Config.DEFAULT_NUM_INFERENCE_STEPS))
        
        logger.info(f"Generating image for prompt: '{prompt[:50]}...'")
        logger.info(f"Parameters - guidance_scale: {guidance_scale}, steps: {num_inference_steps}")
        
        # Check pipeline
        if pipeline is None:
            logger.error("Pipeline not initialized")
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Generate image
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            image = output.images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info("Image generated successfully")
        
        return jsonify({
            'image': image_b64,
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps
        })
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        return jsonify({'error': 'GPU out of memory. Try reducing num_inference_steps'}), 500
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

def setup_ngrok():
    """Setup ngrok tunnel"""
    try:
        ngrok.set_auth_token(Config.NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(5000)
        logger.info(f"ngrok tunnel established at: {public_url}")
        logger.info(f"Public URL: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Failed to setup ngrok: {str(e)}")
        logger.info("Continuing without ngrok. Application will only be accessible locally.")
        return None

def main():
    """Main entry point"""
    try:
        # Initialize pipeline
        logger.info("Starting application...")
        initialize_pipeline()
        
        # Setup ngrok
        public_url = setup_ngrok()
        
        # Run Flask app
        logger.info("Starting Flask server on port 5000...")
        if public_url:
            logger.info(f"Access your API at: {public_url}/generate")
        logger.info("Local access: http://localhost:5000")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == '__main__':
    main()