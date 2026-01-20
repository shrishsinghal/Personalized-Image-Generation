# Training Guide

This guide covers training your own LoRA (Low-Rank Adaptation) fine-tuned model for Stable Diffusion XL.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Running Training](#running-training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Trains only ~1% of model parameters
- Requires significantly less memory and compute
- Enables fast training (2-4 hours on consumer GPUs)
- Produces small, shareable weight files (~25MB)

## Requirements

### Hardware

**Minimum:**
- GPU: NVIDIA GPU with 12GB+ VRAM (RTX 3060 12GB, T4, etc.)
- RAM: 16GB system RAM
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA GPU with 16GB+ VRAM (A100, RTX 4090, etc.)
- RAM: 32GB system RAM
- Storage: 100GB free space (for datasets and checkpoints)

### Software

```bash
pip install -r training/requirements.txt
```

## Dataset Preparation

### 1. Collect Images

Collect 10-100 high-quality images for your target domain. For best results:
- Use consistent style/subject matter
- Minimum resolution: 512x512 pixels
- High-quality, well-composed images
- Diverse angles and compositions

### 2. Organize Dataset

```
my_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── metadata.jsonl
```

### 3. Create Captions

Create `metadata.jsonl` with image captions:

```jsonl
{"file_name": "image_001.jpg", "text": "A detailed caption describing the image"}
{"file_name": "image_002.jpg", "text": "Another detailed caption"}
```

**Caption Guidelines:**
- Be specific and descriptive
- Include important details (colors, objects, actions)
- Mention artistic style if relevant
- Keep length: 10-75 words

### 4. Upload to Hugging Face Hub

```python
from datasets import Dataset
import pandas as pd

# Create dataset
df = pd.read_json('metadata.jsonl', lines=True)
dataset = Dataset.from_pandas(df)

# Upload
dataset.push_to_hub("your-username/your-dataset")
```

## Training Configuration

### Basic Configuration

Create `training_config.yaml`:

```yaml
# Model settings
model_name: "stabilityai/stable-diffusion-xl-base-1.0"
vae_name: "madebyollin/sdxl-vae-fp16-fix"

# Dataset
dataset_name: "your-username/your-dataset"
caption_column: "text"
image_column: "file_name"

# LoRA settings
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules:
  - "to_k"
  - "to_q"
  - "to_v"
  - "to_out.0"

# Training hyperparameters
learning_rate: 1e-4
num_train_epochs: 100
train_batch_size: 1
gradient_accumulation_steps: 4
mixed_precision: "fp16"

# Output
output_dir: "./outputs"
save_steps: 500
save_total_limit: 5
```

### Key Hyperparameters Explained

**LoRA Rank (r)**
- Controls the dimensionality of LoRA matrices
- Higher = more capacity, but more parameters
- Typical range: 4-16
- **Recommendation:** Start with 8

**LoRA Alpha**
- Scaling factor for LoRA weights
- Typically set to 2×rank or 1×rank
- **Recommendation:** Use 16 (2×rank) for rank=8

**Learning Rate**
- Controls how quickly the model adapts
- Too high: unstable training
- Too low: slow convergence
- **Recommendation:** 1e-4 for SDXL

**Batch Size**
- Number of samples per gradient update
- Limited by GPU memory
- **Recommendation:** 1 with gradient_accumulation_steps=4

## Running Training

### Using Training Script

```bash
python training/train_lora.py \
  --config training_config.yaml \
  --hub_model_id "your-username/your-lora-model"
```

### Using Accelerate (Recommended)

```bash
accelerate launch --mixed_precision="fp16" \
  training/train_lora.py \
  --config training_config.yaml \
  --hub_model_id "your-username/your-lora-model"
```

### Monitoring Training

Training logs will show:
- Loss values (should decrease over time)
- Learning rate
- Training speed (iterations/second)
- Estimated time remaining

Expected training times:
- 10 images, 100 epochs: ~30 minutes
- 50 images, 100 epochs: ~2 hours
- 100 images, 100 epochs: ~4 hours

### Checkpointing

Models are saved every `save_steps` iterations to:
```
outputs/
├── checkpoint-500/
├── checkpoint-1000/
└── ...
```

## Evaluation

### Quantitative Evaluation

Run comparative analysis to get CLIP scores:

```bash
python training/comparative_analysis.py
```

This will:
1. Generate images using your model
2. Calculate CLIP scores
3. Compare against baseline models
4. Generate visualization plots

### Qualitative Evaluation

Generate test images:

```python
from diffusers import DiffusionPipeline
import torch

# Load your model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("your-username/your-lora-model")
pipe = pipe.to("cuda")

# Test prompts
prompts = [
    "Test prompt 1",
    "Test prompt 2",
    # Add more test prompts
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f"test_{i}.png")
```

### Evaluation Criteria

Good LoRA models should:
- Generate coherent, high-quality images
- Follow prompts accurately (high CLIP score)
- Capture the target style/domain
- Maintain diversity across generations

## Advanced Techniques

### Multi-Concept Training

Train on multiple concepts by organizing images into subdirectories:

```
dataset/
├── concept_1/
│   └── images...
└── concept_2/
    └── images...
```

### Prompt Engineering for Training

Better captions = better results:
- Use consistent terminology
- Include trigger words for specific styles
- Describe both content and style
- Example: "A [subject] in [your style], detailed, high quality"

### Hyperparameter Tuning

Experiment with:
- **Higher rank (16-32)**: More capacity for complex styles
- **Lower learning rate (5e-5)**: More stable training
- **More epochs (150-200)**: Better convergence for small datasets
- **Different target modules**: Focus on specific model components

## Troubleshooting

### Out of Memory

**Solutions:**
- Reduce `train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use `--gradient_checkpointing`
- Reduce LoRA rank

### Poor Quality Results

**Solutions:**
- Increase training epochs
- Improve caption quality
- Add more training images
- Increase LoRA rank
- Adjust learning rate

### Overfitting

**Symptoms:**
- Images look identical to training data
- Low diversity in generations

**Solutions:**
- Reduce training epochs
- Add more diverse training data
- Increase LoRA dropout
- Use data augmentation

### Training Not Converging

**Solutions:**
- Check learning rate (try 5e-5 or 2e-4)
- Verify dataset quality
- Ensure captions are accurate
- Try different random seed

## Best Practices

1. **Start Small**: Test with 10-20 images before scaling up
2. **Good Data > More Data**: Quality matters more than quantity
3. **Monitor Loss**: Should decrease steadily (some fluctuation is normal)
4. **Save Checkpoints**: Keep multiple checkpoints to find the best
5. **Test Early**: Generate samples every few hundred steps
6. **Compare**: Always compare against base model

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Dataset Preparation Guide](https://huggingface.co/docs/datasets/image_dataset)

## Next Steps

After training:
1. Push your model to Hugging Face Hub
2. Test with the web interface
3. Share your results with the community
4. Consider training on larger/more diverse datasets

---

Need help? Open an issue on GitHub or check the discussions forum!