# Fine-Tuned Stable Diffusion XL with LoRA

A production-ready implementation of Stable Diffusion XL fine-tuned using Low-Rank Adaptation (LoRA) for personalized image generation. This project includes comprehensive training pipelines, quantitative evaluation metrics, and a user-friendly web interface.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project demonstrates the complete pipeline for fine-tuning Stable Diffusion XL using LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique. The implementation includes:

- **LoRA Fine-tuning**: Efficient model adaptation with minimal trainable parameters
- **Comparative Analysis**: Quantitative evaluation using CLIP scores across multiple models
- **Production Backend**: Flask API for scalable image generation
- **Interactive Frontend**: Gradio-based web interface with real-time generation
- **Quality Metrics**: CLIP score evaluation for text-image alignment

## 🏗️ Architecture

```
┌─────────────────┐
│  User Interface │  ← Gradio Web App
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Flask API     │  ← Image Generation Endpoint
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  SDXL + LoRA    │  ← Fine-tuned Model
└─────────────────┘
```

## 📊 Key Results

| Model | Average CLIP Score | Training Time | Parameters Fine-tuned |
|-------|-------------------|---------------|----------------------|
| Stable Diffusion v1.5 | 28.34 | Baseline | Full Model |
| DALL-E Mini | 24.67 | N/A | N/A |
| **SDXL + LoRA (Ours)** | **31.45** | ~2 hours | <1% of base model |

## 🚀 Quick Start

### Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/sdxl-lora-finetuning.git
cd sdxl-lora-finetuning

# Run automated setup script
chmod +x scripts/quickstart.sh
./scripts/quickstart.sh

# Test installation
./test_setup.sh
```

### Manual Setup

#### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (12GB+ VRAM recommended)
- ngrok account (for backend deployment)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sdxl-lora-finetuning.git
cd sdxl-lora-finetuning
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# For backend
pip install -r backend/requirements.txt

# For frontend
pip install -r frontend/requirements.txt

# For training (optional)
pip install -r training/requirements.txt
```

### Running the Application

#### Backend (Flask API)

1. Configure ngrok authentication:
```bash
ngrok config add-authtoken YOUR_NGROK_TOKEN
```

2. Start the backend:
```bash
cd backend
python app.py
```

The backend will start and display your ngrok public URL. Copy this URL.

#### Frontend (Gradio Interface)

1. Update the ngrok URL in `frontend/web_app.py`:
```python
url = "YOUR_NGROK_URL/generate"
```

2. Launch the frontend:
```bash
cd frontend
python web_app.py
```

3. Open the provided local URL in your browser (typically `http://127.0.0.1:7860`)

## 📁 Project Structure

```
sdxl-lora-finetuning/
│
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Backend dependencies
│   └── utils.py              # Helper functions
│
├── frontend/
│   ├── web_app.py            # Gradio interface
│   └── requirements.txt      # Frontend dependencies
│
├── training/
│   ├── comparative_analysis.py    # Model evaluation
│   ├── sdxl_lora_training.py     # LoRA fine-tuning
│   └── requirements.txt          # Training dependencies
│
├── notebooks/                # Jupyter notebooks for experiments
│   ├── DL_comparative_analysis.ipynb
│   └── dl_sdxl_dreambooth.ipynb
│
├── docs/
│   ├── TRAINING.md          # Training guide
│   ├── API.md               # API documentation
│   └── EVALUATION.md        # Evaluation methodology
│
├── .gitignore
├── LICENSE
└── README.md
```

## 🎓 Training Your Own Model

See [TRAINING.md](docs/TRAINING.md) for detailed instructions on:
- Preparing your dataset
- Configuring LoRA parameters
- Training the model
- Evaluating results

## 📊 Evaluation Methodology

We use CLIP (Contrastive Language-Image Pre-training) scores to quantitatively evaluate text-image alignment:

```python
# Calculate CLIP score for a generated image
clip_score = evaluate_clip_score(generated_image, text_prompt)
```

Higher CLIP scores indicate better alignment between the generated image and the input text. See [EVALUATION.md](docs/EVALUATION.md) for more details.

## 🔧 API Reference

### Generate Image Endpoint

**POST** `/generate`

```json
{
  "prompt": "A golden retriever playing in the snow",
  "guidance_scale": 7.5,
  "num_inference_steps": 50
}
```

**Response:**
```json
{
  "image": "base64_encoded_image_data"
}
```

See [API.md](docs/API.md) for complete API documentation.

## 🛠️ Technical Details

### LoRA Fine-tuning

Low-Rank Adaptation (LoRA) enables efficient fine-tuning by:
- Adding trainable low-rank matrices to model layers
- Keeping base model weights frozen
- Reducing trainable parameters by 99%+
- Enabling fast training and easy model sharing

**Key Parameters:**
- Rank (r): 8
- Alpha: 16
- Dropout: 0.1
- Target modules: attention layers

### Model Architecture

- **Base Model**: Stable Diffusion XL 1.0
- **VAE**: SDXL VAE FP16 (madebyollin/sdxl-vae-fp16-fix)
- **Text Encoders**: CLIP ViT-L/14 + OpenCLIP ViT-bigG/14
- **LoRA Weights**: Custom trained on curated dataset

## 📈 Performance Benchmarks

Tested on NVIDIA T4 GPU:

| Operation | Time | Memory |
|-----------|------|--------|
| Single image generation | ~40s | 10.2 GB |
| Batch of 5 images | ~2.5min | 11.8 GB |
| Model loading | ~15s | 8.5 GB |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) by Stability AI
- [LoRA](https://github.com/microsoft/LoRA) by Microsoft Research
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [CLIP](https://github.com/openai/CLIP) by OpenAI

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/sdxl-lora-finetuning](https://github.com/yourusername/sdxl-lora-finetuning)

## 🔗 Resources

- [Research Paper on LoRA](https://arxiv.org/abs/2106.09685)
- [Stable Diffusion XL Paper](https://arxiv.org/abs/2307.01952)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Project Blog Post](your-blog-link)

---

⭐ If you found this project helpful, please consider giving it a star!