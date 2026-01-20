"""
Production LoRA Training Script for Stable Diffusion XL
Fine-tune SDXL using DreamBooth + LoRA on custom datasets
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from huggingface_hub import HfApi, create_repo, upload_folder
from accelerate import Accelerator
from accelerate.utils import set_seed
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """Handles the complete LoRA training pipeline"""
    
    def __init__(self, config: argparse.Namespace):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        logger.info(f"Initialized trainer on device: {self.device}")
        
    def prepare_images(self, source_dir: str, dest_dir: str) -> List[str]:
        """
        Copy and organize images for training
        
        Args:
            source_dir: Source directory containing images
            dest_dir: Destination directory
            
        Returns:
            List of image paths
        """
        logger.info(f"Preparing images from {source_dir}")
        
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy images
        if os.path.isdir(source_dir):
            for file in os.listdir(source_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(source_dir, file)
                    dst = os.path.join(dest_dir, file)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
        
        # Get image paths
        image_paths = [
            os.path.join(dest_dir, f) for f in os.listdir(dest_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        logger.info(f"Prepared {len(image_paths)} images")
        return image_paths
    
    def generate_captions(
        self,
        image_paths: List[str],
        caption_prefix: str = "",
        output_path: Optional[str] = None
    ) -> dict:
        """
        Generate captions for images using BLIP
        
        Args:
            image_paths: List of image paths
            caption_prefix: Prefix to add to captions
            output_path: Optional path to save metadata
            
        Returns:
            Dictionary mapping filenames to captions
        """
        logger.info("Loading BLIP model for captioning...")
        
        # Load BLIP model
        processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16
        ).to(self.device)
        
        captions = {}
        
        logger.info(f"Generating captions for {len(image_paths)} images...")
        for img_path in image_paths:
            try:
                # Load and process image
                image = Image.open(img_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt").to(
                    self.device, torch.float16
                )
                
                # Generate caption
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=50
                )
                caption = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                # Add prefix
                full_caption = f"{caption_prefix} {caption}".strip()
                filename = os.path.basename(img_path)
                captions[filename] = full_caption
                
                logger.info(f"{filename}: {full_caption}")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
        
        # Save metadata if path provided
        if output_path:
            self._save_metadata(captions, output_path)
        
        # Clean up BLIP model
        del processor, model
        torch.cuda.empty_cache()
        
        return captions
    
    def _save_metadata(self, captions: dict, output_path: str):
        """Save captions as metadata.jsonl"""
        import json
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for filename, caption in captions.items():
                entry = {
                    "file_name": filename,
                    "prompt": caption
                }
                json.dump(entry, f)
                f.write('\n')
        
        logger.info(f"Saved metadata to {output_path}")
    
    def train(self):
        """Execute LoRA training"""
        logger.info("Starting LoRA training...")
        logger.info(f"Configuration: {vars(self.config)}")
        
        # Build training command
        cmd = self._build_training_command()
        
        # Log command
        logger.info(f"Training command: {' '.join(cmd)}")
        
        # Execute training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Training completed successfully!")
            logger.info(result.stdout)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with error: {e}")
            logger.error(f"Output: {e.output}")
            logger.error(f"Error: {e.stderr}")
            raise
    
    def _build_training_command(self) -> List[str]:
        """Build the accelerate launch command"""
        script_path = os.path.join(
            os.path.dirname(__file__),
            "train_dreambooth_lora_sdxl.py"
        )
        
        # Download training script if it doesn't exist
        if not os.path.exists(script_path):
            logger.info("Downloading training script...")
            import urllib.request
            url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py"
            urllib.request.urlretrieve(url, script_path)
        
        cmd = [
            "accelerate", "launch",
            script_path,
            f"--pretrained_model_name_or_path={self.config.model_name}",
            f"--pretrained_vae_model_name_or_path={self.config.vae_name}",
            f"--instance_data_dir={self.config.instance_data_dir}",
            f"--output_dir={self.config.output_dir}",
            f"--mixed_precision={self.config.mixed_precision}",
            f"--instance_prompt={self.config.instance_prompt}",
            f"--resolution={self.config.resolution}",
            f"--train_batch_size={self.config.train_batch_size}",
            f"--gradient_accumulation_steps={self.config.gradient_accumulation_steps}",
            f"--learning_rate={self.config.learning_rate}",
            f"--lr_scheduler={self.config.lr_scheduler}",
            f"--lr_warmup_steps={self.config.lr_warmup_steps}",
            f"--max_train_steps={self.config.max_train_steps}",
            f"--checkpointing_steps={self.config.checkpointing_steps}",
            f"--seed={self.config.seed}",
        ]
        
        # Optional arguments
        if self.config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        
        if self.config.use_8bit_adam:
            cmd.append("--use_8bit_adam")
        
        if self.config.snr_gamma:
            cmd.append(f"--snr_gamma={self.config.snr_gamma}")
        
        return cmd
    
    def push_to_hub(self, repo_id: str):
        """
        Push trained model to Hugging Face Hub
        
        Args:
            repo_id: Repository ID (username/model-name)
        """
        logger.info(f"Pushing model to Hub: {repo_id}")
        
        try:
            # Create repository
            api = HfApi()
            create_repo(repo_id, exist_ok=True, repo_type="model")
            
            # Save model card
            self._save_model_card(repo_id)
            
            # Upload folder
            api.upload_folder(
                folder_path=self.config.output_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Training completed",
                ignore_patterns=["step_*", "epoch_*", "*.bin"]
            )
            
            logger.info(f"Model pushed successfully to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to push to hub: {str(e)}")
            raise
    
    def _save_model_card(self, repo_id: str):
        """Generate and save model card"""
        model_card = f"""---
license: openrail++
base_model: {self.config.model_name}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
- dreambooth
instance_prompt: {self.config.instance_prompt}
---

# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {self.config.model_name}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [diffusers trainer](https://github.com/huggingface/diffusers).

## Training Configuration

- **Base Model**: {self.config.model_name}
- **VAE**: {self.config.vae_name}
- **Instance Prompt**: `{self.config.instance_prompt}`
- **Resolution**: {self.config.resolution}
- **Training Steps**: {self.config.max_train_steps}
- **Learning Rate**: {self.config.learning_rate}
- **Batch Size**: {self.config.train_batch_size}
- **Gradient Accumulation**: {self.config.gradient_accumulation_steps}

## Usage

```python
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "{self.config.vae_name}",
    torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "{self.config.model_name}",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

pipe.load_lora_weights("{repo_id}")
pipe = pipe.to("cuda")

prompt = "{self.config.instance_prompt}"
image = pipe(prompt, num_inference_steps=50).images[0]
image.save("output.png")
```

## License

This model is licensed under the [SDXL License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).
"""
        
        card_path = os.path.join(self.config.output_dir, "README.md")
        with open(card_path, 'w') as f:
            f.write(model_card)
        
        logger.info(f"Model card saved to {card_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train LoRA for Stable Diffusion XL"
    )
    
    # Data arguments
    parser.add_argument(
        "--source_images_dir",
        type=str,
        required=True,
        help="Directory containing source images"
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="./training_images",
        help="Directory where processed images will be stored"
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="The prompt describing the instance (e.g., 'A photo of [person] wearing casual clothes')"
    )
    parser.add_argument(
        "--caption_prefix",
        type=str,
        default="",
        help="Prefix to add to auto-generated captions"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--vae_name",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Pretrained VAE name or path"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training image resolution"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Learning rate scheduler"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer"
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR gamma for loss weighting"
    )
    
    # Hub arguments
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub model ID (username/model-name)"
    )
    
    # Other arguments
    parser.add_argument(
        "--skip_caption_generation",
        action="store_true",
        help="Skip automatic caption generation (use existing metadata)"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize trainer
    trainer = LoRATrainer(args)
    
    # Step 1: Prepare images
    logger.info("="*50)
    logger.info("STEP 1: Preparing Images")
    logger.info("="*50)
    
    image_paths = trainer.prepare_images(
        args.source_images_dir,
        args.instance_data_dir
    )
    
    if len(image_paths) == 0:
        logger.error("No images found! Please check your source directory.")
        return
    
    # Step 2: Generate captions
    if not args.skip_caption_generation:
        logger.info("="*50)
        logger.info("STEP 2: Generating Captions")
        logger.info("="*50)
        
        metadata_path = os.path.join(
            os.path.dirname(args.instance_data_dir),
            "metadata.jsonl"
        )
        
        captions = trainer.generate_captions(
            image_paths,
            caption_prefix=args.caption_prefix or args.instance_prompt,
            output_path=metadata_path
        )
    else:
        logger.info("Skipping caption generation")
    
    # Step 3: Train
    logger.info("="*50)
    logger.info("STEP 3: Training LoRA")
    logger.info("="*50)
    
    trainer.train()
    
    # Step 4: Push to Hub (optional)
    if args.push_to_hub:
        logger.info("="*50)
        logger.info("STEP 4: Pushing to Hub")
        logger.info("="*50)
        
        if not args.hub_model_id:
            logger.error("--hub_model_id required when using --push_to_hub")
            return
        
        trainer.push_to_hub(args.hub_model_id)
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*50)
    logger.info(f"Model saved to: {args.output_dir}")
    if args.push_to_hub:
        logger.info(f"Model available at: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()