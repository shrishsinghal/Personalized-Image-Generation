# Usage Examples

Complete examples for training, evaluation, and deployment.

## Table of Contents

- [Training Examples](#training-examples)
- [Inference Examples](#inference-examples)
- [Evaluation Examples](#evaluation-examples)
- [API Examples](#api-examples)

## Training Examples

### Example 1: Basic Training

Train a LoRA model on a custom dataset:

```bash
python training/train_lora.py \
  --source_images_dir ./my_photos \
  --instance_prompt "A photo of sks person" \
  --output_dir ./my_lora_model \
  --max_train_steps 500 \
  --push_to_hub \
  --hub_model_id "username/my-lora-model"
```

**What this does:**
- Loads images from `./my_photos`
- Auto-generates captions with BLIP
- Trains for 500 steps
- Pushes to Hugging Face Hub

### Example 2: Advanced Training

Fine-tuned training with custom parameters:

```bash
python training/train_lora.py \
  --source_images_dir ./dataset/images \
  --instance_prompt "A photo of [subject] in [style]" \
  --caption_prefix "Professional photograph of" \
  --output_dir ./advanced_lora \
  --resolution 1024 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine \
  --max_train_steps 1000 \
  --checkpointing_steps 250 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --snr_gamma 5.0 \
  --seed 42
```

**Advanced features:**
- Custom caption prefix
- Cosine learning rate schedule
- More training steps
- Gradient checkpointing for memory efficiency
- 8-bit Adam optimizer
- Reproducible with seed

### Example 3: Portrait/Person Training

Train a model on portrait photos:

```bash
python training/train_lora.py \
  --source_images_dir ./portraits \
  --instance_prompt "A professional headshot of John Doe" \
  --caption_prefix "A portrait photograph showing" \
  --output_dir ./portrait_lora \
  --resolution 1024 \
  --max_train_steps 750 \
  --learning_rate 1e-4 \
  --seed 123 \
  --push_to_hub \
  --hub_model_id "username/portrait-lora"
```

### Example 4: Style Training

Train on a specific artistic style:

```bash
python training/train_lora.py \
  --source_images_dir ./art_style_examples \
  --instance_prompt "An artwork in the style of [artist]" \
  --caption_prefix "A digital artwork featuring" \
  --output_dir ./style_lora \
  --max_train_steps 1500 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 4 \
  --checkpointing_steps 300
```

### Example 5: Resume from Checkpoint

Continue training from a previous checkpoint:

```bash
# First, identify the checkpoint directory
# e.g., ./my_lora_model/checkpoint-500

python training/train_lora.py \
  --source_images_dir ./my_photos \
  --instance_prompt "A photo of sks person" \
  --output_dir ./my_lora_model \
  --resume_from_checkpoint ./my_lora_model/checkpoint-500 \
  --max_train_steps 1000
```

## Inference Examples

### Example 1: Basic Image Generation

Generate a single image:

```bash
python training/inference.py \
  --lora_weights "username/my-lora-model" \
  --prompt "A photo of sks person at the beach, sunset, golden hour" \
  --output_dir ./outputs
```

### Example 2: Multiple Images

Generate multiple variations:

```bash
python training/inference.py \
  --lora_weights "username/my-lora-model" \
  --prompt "A photo of sks person wearing professional attire" \
  --num_images 4 \
  --save_grid \
  --output_dir ./outputs \
  --output_prefix "professional"
```

**Output:**
- `professional_1.png` through `professional_4.png`
- `professional_grid.png` (4 images in a grid)

### Example 3: High Quality Generation

Generate with more inference steps:

```bash
python training/inference.py \
  --lora_weights "username/portrait-lora" \
  --prompt "A professional headshot of John Doe, studio lighting, grey background" \
  --num_inference_steps 100 \
  --guidance_scale 8.0 \
  --seed 42 \
  --output_dir ./high_quality
```

**Parameters explained:**
- `num_inference_steps=100`: More steps = better quality
- `guidance_scale=8.0`: Higher adherence to prompt
- `seed=42`: Reproducible results

### Example 4: Batch Generation from File

Create `prompts.txt`:
```
A photo of sks person at a coffee shop
A photo of sks person hiking in mountains
A photo of sks person in business meeting
A photo of sks person at sunset beach
```

Generate all:
```bash
python training/inference.py \
  --lora_weights "username/my-lora-model" \
  --prompts_file prompts.txt \
  --num_images 2 \
  --output_dir ./batch_outputs
```

### Example 5: Custom Resolution

Generate at different resolutions:

```bash
# Portrait orientation
python training/inference.py \
  --lora_weights "username/my-lora-model" \
  --prompt "A full body photo of sks person" \
  --height 1280 \
  --width 768 \
  --output_dir ./portrait_images

# Landscape orientation
python training/inference.py \
  --lora_weights "username/my-lora-model" \
  --prompt "A wide shot of sks person in nature" \
  --height 768 \
  --width 1280 \
  --output_dir ./landscape_images
```

### Example 6: Using Negative Prompts

Improve quality with negative prompts:

```bash
python training/inference.py \
  --lora_weights "username/my-lora-model" \
  --prompt "A photo of sks person, professional, high quality" \
  --negative_prompt "blurry, low quality, distorted, ugly, bad anatomy" \
  --num_inference_steps 75 \
  --guidance_scale 9.0 \
  --output_dir ./outputs
```

## Evaluation Examples

### Example 1: Compare Models

Evaluate multiple models:

```bash
python training/comparative_analysis.py
```

**Output:**
- `model_comparison.png`: Visual comparison
- `average_scores.png`: CLIP score bar chart
- Console output with detailed scores

### Example 2: Custom Test Prompts

Edit `comparative_analysis.py` to test specific prompts:

```python
EVALUATION_PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    "Your custom prompt 3",
]
```

Then run:
```bash
python training/comparative_analysis.py
```

## API Examples

### Example 1: Python Requests

```python
import requests
import base64
from PIL import Image
import io

# Generate image
url = "http://localhost:5000/generate"
payload = {
    "prompt": "A photo of sks person at the beach",
    "guidance_scale": 7.5,
    "num_inference_steps": 50
}

response = requests.post(url, json=payload, timeout=120)
data = response.json()

# Decode and save
image_data = base64.b64decode(data['image'])
image = Image.open(io.BytesIO(image_data))
image.save('output.png')
print("Image saved!")
```

### Example 2: cURL

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photo of sks person wearing sunglasses",
    "guidance_scale": 8.0,
    "num_inference_steps": 60
  }' \
  --output response.json

# Extract image from response
python -c "
import json
import base64
with open('response.json') as f:
    data = json.load(f)
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(data['image']))
"
```

### Example 3: JavaScript/Node.js

```javascript
const fetch = require('node-fetch');
const fs = require('fs');

async function generateImage() {
  const response = await fetch('http://localhost:5000/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: 'A photo of sks person in winter',
      guidance_scale: 7.5,
      num_inference_steps: 50
    })
  });
  
  const data = await response.json();
  
  // Save image
  const buffer = Buffer.from(data.image, 'base64');
  fs.writeFileSync('output.png', buffer);
  
  console.log('Image saved!');
}

generateImage();
```

### Example 4: Batch Processing

```python
import requests
import base64
from PIL import Image
import io
from pathlib import Path

def generate_and_save(prompt, output_path):
    """Generate image and save to path"""
    response = requests.post(
        "http://localhost:5000/generate",
        json={
            "prompt": prompt,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        },
        timeout=120
    )
    
    data = response.json()
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    print(f"Saved: {output_path}")

# Batch generate
prompts = [
    "A photo of sks person at sunrise",
    "A photo of sks person in city",
    "A photo of sks person reading book",
]

output_dir = Path("./batch_api_outputs")
output_dir.mkdir(exist_ok=True)

for i, prompt in enumerate(prompts):
    output_path = output_dir / f"image_{i+1}.png"
    generate_and_save(prompt, output_path)

print(f"Generated {len(prompts)} images!")
```

## Complete Workflow Example

Here's a complete workflow from training to deployment:

### Step 1: Prepare Dataset

```bash
# Organize your images
mkdir -p ./my_dataset
cp /path/to/photos/* ./my_dataset/
```

### Step 2: Train Model

```bash
python training/train_lora.py \
  --source_images_dir ./my_dataset \
  --instance_prompt "A photo of sks person" \
  --output_dir ./trained_model \
  --max_train_steps 500 \
  --push_to_hub \
  --hub_model_id "username/my-awesome-lora"
```

Wait ~2 hours for training to complete.

### Step 3: Test Locally

```bash
python training/inference.py \
  --lora_weights "username/my-awesome-lora" \
  --prompt "A photo of sks person smiling" \
  --num_images 4 \
  --save_grid \
  --output_dir ./test_outputs
```

### Step 4: Evaluate Quality

```bash
# Edit comparative_analysis.py to use your model
python training/comparative_analysis.py
```

Review `model_comparison.png` and scores.

### Step 5: Deploy Backend

**Terminal 1:**
```bash
cd backend
python app.py
```

Note the ngrok URL displayed.

### Step 6: Start Frontend

**Terminal 2:**
```bash
cd frontend
# Edit web_app.py to use your model ID and ngrok URL
python web_app.py
```

### Step 7: Use the Interface

Open browser to `http://127.0.0.1:7860` and start generating!

## Tips and Tricks

### Getting Better Results

**1. Training:**
- Use 10-20 high-quality images minimum
- Write detailed, consistent captions
- Train for 500-1000 steps
- Use gradient checkpointing if memory limited

**2. Inference:**
- Start with default settings
- Increase steps (75-100) for quality
- Adjust guidance_scale:
  - 7-8: Balanced
  - 9-12: More literal to prompt
  - 5-6: More creative
- Use negative prompts to avoid artifacts

**3. Prompts:**
- Be specific and detailed
- Include style keywords
- Mention lighting, composition
- Example: "A professional photograph of [subject], studio lighting, high quality, detailed"

### Common Issues

**Training too slow:**
```bash
# Use 8-bit Adam and gradient checkpointing
--use_8bit_adam --gradient_checkpointing
```

**Out of memory:**
```bash
# Reduce batch size
--train_batch_size 1 --gradient_accumulation_steps 4
```

**Poor quality results:**
```bash
# More steps and better prompts
--max_train_steps 1000
# Better instance prompt
--instance_prompt "A high quality photo of [subject]"
```

---

Need more examples? Check the [discussions](https://github.com/yourusername/sdxl-lora-finetuning/discussions) or open an issue!