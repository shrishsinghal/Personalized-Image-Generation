"""
Inference script for LoRA fine-tuned SDXL models
Generate images using trained LoRA weights
"""

import os
import argparse
import logging
from typing import List, Optional
from pathlib import Path

import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """
    Create a grid of images
    
    Args:
        images: List of PIL Images
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        PIL Image grid
    """
    if len(images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(images)}")
    
    # Get dimensions from first image
    w, h = images[0].size
    
    # Create grid
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, box=(col * w, row * h))
    
    return grid


class LoRAInference:
    """Handles inference with LoRA fine-tuned models"""
    
    def __init__(
        self,
        lora_weights: str,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
        device: str = None
    ):
        """
        Initialize inference pipeline
        
        Args:
            lora_weights: Path or Hub ID of LoRA weights
            base_model: Base SDXL model
            vae_model: VAE model
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing pipeline on {self.device}")
        
        # Load VAE
        logger.info(f"Loading VAE: {vae_model}")
        vae = AutoencoderKL.from_pretrained(
            vae_model,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        
        # Load base model
        logger.info(f"Loading base model: {base_model}")
        self.pipe = DiffusionPipeline.from_pretrained(
            base_model,
            vae=vae,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            variant="fp16" if self.device == 'cuda' else None,
            use_safetensors=True
        )
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights: {lora_weights}")
        self.pipe.load_lora_weights(lora_weights)
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable optimizations
        if self.device == 'cuda':
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
        
        logger.info("Pipeline initialized successfully!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images from prompt
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt (optional)
            num_images: Number of images to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width
            seed: Random seed (optional)
            
        Returns:
            List of generated PIL Images
        """
        logger.info(f"Generating {num_images} image(s)")
        logger.info(f"Prompt: {prompt}")
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Using seed: {seed}")
        
        # Generate
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            )
        
        return output.images
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[Image.Image]]:
        """
        Generate images for multiple prompts
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List of image lists (one per prompt)
        """
        all_images = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            images = self.generate(prompt, **kwargs)
            all_images.append(images)
        
        return all_images


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate images with LoRA fine-tuned SDXL"
    )
    
    # Model arguments
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="LoRA weights (path or Hub ID)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model"
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="VAE model"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Output directory"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="generated",
        help="Output filename prefix"
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save images as grid if multiple images"
    )
    
    # Batch generation
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Text file with prompts (one per line)"
    )
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = LoRAInference(
        lora_weights=args.lora_weights,
        base_model=args.base_model,
        vae_model=args.vae_model
    )
    
    # Determine prompts
    if args.prompts_file:
        logger.info(f"Loading prompts from {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts")
    else:
        prompts = [args.prompt]
    
    # Generate images
    logger.info("="*50)
    logger.info("GENERATING IMAGES")
    logger.info("="*50)
    
    all_images = []
    
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt}")
        
        images = pipeline.generate(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            seed=args.seed
        )
        
        # Save individual images
        for img_idx, image in enumerate(images):
            if len(prompts) > 1:
                filename = f"{args.output_prefix}_prompt{prompt_idx+1}_img{img_idx+1}.png"
            else:
                filename = f"{args.output_prefix}_{img_idx+1}.png"
            
            output_path = os.path.join(args.output_dir, filename)
            image.save(output_path)
            logger.info(f"Saved: {output_path}")
        
        all_images.extend(images)
        
        # Save grid if requested
        if args.save_grid and args.num_images > 1:
            # Determine grid dimensions
            cols = min(args.num_images, 3)
            rows = (args.num_images + cols - 1) // cols
            
            grid = create_image_grid(images, rows, cols)
            
            if len(prompts) > 1:
                grid_filename = f"{args.output_prefix}_prompt{prompt_idx+1}_grid.png"
            else:
                grid_filename = f"{args.output_prefix}_grid.png"
            
            grid_path = os.path.join(args.output_dir, grid_filename)
            grid.save(grid_path)
            logger.info(f"Saved grid: {grid_path}")
    
    # Summary
    logger.info("="*50)
    logger.info("GENERATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Total images generated: {len(all_images)}")
    logger.info(f"Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()