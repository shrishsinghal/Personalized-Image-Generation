"""
Comparative Analysis of Text-to-Image Models
Evaluates multiple models using CLIP scores for quantitative comparison
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoencoderKL
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test prompts for evaluation
EVALUATION_PROMPTS = [
    "A golden retriever playing in the snow.",
    "A sunset over a mountain range.",
    "A cat sitting on a couch.",
    "An astronaut riding a horse on mars",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

class ModelEvaluator:
    """Evaluates text-to-image models using CLIP scores"""
    
    def __init__(self, device: str = None):
        """
        Initialize evaluator
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing evaluator on {self.device}")
        
        # Load CLIP model for evaluation
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        
    def calculate_clip_score(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> List[float]:
        """
        Calculate CLIP scores for image-text pairs
        
        Args:
            images: List of PIL Images
            prompts: List of corresponding text prompts
            
        Returns:
            List of CLIP scores
        """
        scores = []
        
        for image, prompt in zip(images, prompts):
            # Prepare inputs
            inputs = self.clip_processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Calculate score
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                score = outputs.logits_per_image.item()
                scores.append(score)
        
        return scores
    
    def evaluate_model(
        self,
        model_name: str,
        pipeline,
        prompts: List[str] = None
    ) -> Tuple[List[Image.Image], List[float], float]:
        """
        Evaluate a model on test prompts
        
        Args:
            model_name: Name of the model being evaluated
            pipeline: Initialized diffusion pipeline
            prompts: List of prompts (uses default if None)
            
        Returns:
            Tuple of (images, scores, average_score)
        """
        prompts = prompts or EVALUATION_PROMPTS
        logger.info(f"Evaluating {model_name} on {len(prompts)} prompts")
        
        images = []
        
        # Generate images
        for prompt in tqdm(prompts, desc=f"Generating with {model_name}"):
            with torch.inference_mode():
                output = pipeline(prompt, num_inference_steps=50)
                images.append(output.images[0])
        
        # Calculate CLIP scores
        scores = self.calculate_clip_score(images, prompts)
        average_score = np.mean(scores)
        
        logger.info(f"{model_name} - Average CLIP Score: {average_score:.4f}")
        
        return images, scores, average_score


def evaluate_stable_diffusion_v1_5(evaluator: ModelEvaluator):
    """Evaluate Stable Diffusion v1.5"""
    logger.info("Loading Stable Diffusion v1.5...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(evaluator.device)
    
    images, scores, avg_score = evaluator.evaluate_model(
        "Stable Diffusion v1.5",
        pipeline
    )
    
    return {
        'name': 'Stable Diffusion v1.5',
        'images': images,
        'scores': scores,
        'average': avg_score
    }


def evaluate_sdxl_with_lora(evaluator: ModelEvaluator):
    """Evaluate SDXL with LoRA fine-tuning"""
    logger.info("Loading SDXL with LoRA...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
    
    # Load base model
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Load LoRA weights
    pipeline.load_lora_weights("nikhilsoni700/dl_project_LoRA")
    pipeline = pipeline.to(evaluator.device)
    
    images, scores, avg_score = evaluator.evaluate_model(
        "SDXL + LoRA",
        pipeline
    )
    
    return {
        'name': 'SDXL + LoRA',
        'images': images,
        'scores': scores,
        'average': avg_score
    }


def visualize_results(results: List[dict], prompts: List[str]):
    """
    Visualize comparison results
    
    Args:
        results: List of result dictionaries
        prompts: List of prompts used
    """
    n_models = len(results)
    n_prompts = len(prompts)
    
    # Create figure
    fig, axes = plt.subplots(
        n_prompts,
        n_models,
        figsize=(5 * n_models, 5 * n_prompts)
    )
    
    if n_prompts == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot images
    for i, prompt in enumerate(prompts):
        for j, result in enumerate(results):
            ax = axes[i, j]
            ax.imshow(result['images'][i])
            ax.axis('off')
            
            title = f"{result['name']}\n"
            title += f"Prompt: {prompt[:30]}...\n"
            title += f"CLIP: {result['scores'][i]:.2f}"
            ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Saved comparison visualization to model_comparison.png")
    plt.close()
    
    # Plot score comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = [r['name'] for r in results]
    avg_scores = [r['average'] for r in results]
    
    bars = ax.bar(model_names, avg_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(results)])
    ax.set_ylabel('Average CLIP Score', fontsize=12)
    ax.set_title('Model Comparison: Average CLIP Scores', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(avg_scores) * 1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{score:.2f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig('average_scores.png', dpi=150, bbox_inches='tight')
    logger.info("Saved score comparison to average_scores.png")
    plt.close()


def main():
    """Run comparative analysis"""
    logger.info("Starting comparative analysis...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    results = []
    
    # Stable Diffusion v1.5
    logger.info("\n" + "="*50)
    logger.info("Evaluating Stable Diffusion v1.5")
    logger.info("="*50)
    sd_v1_5_results = evaluate_stable_diffusion_v1_5(evaluator)
    results.append(sd_v1_5_results)
    
    # SDXL with LoRA
    logger.info("\n" + "="*50)
    logger.info("Evaluating SDXL with LoRA")
    logger.info("="*50)
    sdxl_lora_results = evaluate_sdxl_with_lora(evaluator)
    results.append(sdxl_lora_results)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    for result in results:
        logger.info(f"\n{result['name']}:")
        logger.info(f"  Average CLIP Score: {result['average']:.4f}")
        for i, (prompt, score) in enumerate(zip(EVALUATION_PROMPTS, result['scores'])):
            logger.info(f"  Prompt {i+1}: {score:.4f} - {prompt[:40]}...")
    
    # Visualize results
    visualize_results(results, EVALUATION_PROMPTS)
    
    logger.info("\nAnalysis complete!")
    logger.info("Results saved to model_comparison.png and average_scores.png")


if __name__ == '__main__':
    main()