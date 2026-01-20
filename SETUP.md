# Complete Setup Guide

This guide will walk you through setting up the entire Stable Diffusion XL + LoRA project from scratch.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum:**
- Operating System: Linux, macOS, or Windows 10/11
- GPU: NVIDIA GPU with 12GB+ VRAM and CUDA support
- RAM: 16GB
- Storage: 50GB free space
- Internet: Stable connection for downloading models

**Recommended:**
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 32GB
- Storage: 100GB free space (SSD preferred)

### Software Dependencies

1. **Python 3.8 or higher**

Check your Python version:
```bash
python --version
# or
python3 --version
```

If you need to install Python:
- **Linux**: `sudo apt-get install python3.10`
- **macOS**: `brew install python@3.10`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

2. **CUDA Toolkit** (for NVIDIA GPUs)

Check CUDA installation:
```bash
nvidia-smi
nvcc --version
```

Install CUDA 11.8 or 12.1:
- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

3. **Git**

```bash
git --version
```

If not installed:
- **Linux**: `sudo apt-get install git`
- **macOS**: `brew install git`
- **Windows**: Download from [git-scm.com](https://git-scm.com/)

4. **ngrok Account** (for backend deployment)

- Sign up at [ngrok.com](https://ngrok.com/)
- Get your auth token from the dashboard

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sdxl-lora-finetuning.git
cd sdxl-lora-finetuning
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note:** This will download ~10GB of packages. It may take 10-30 minutes depending on your internet connection.

### Step 5: Install Frontend Dependencies

```bash
cd ../frontend
pip install -r requirements.txt
```

### Step 6: (Optional) Install Training Dependencies

If you plan to train your own models:

```bash
cd ../training
pip install -r requirements.txt
```

## Configuration

### Backend Configuration

1. **Create environment file:**

```bash
cd backend
cp .env.example .env  # Create this if it doesn't exist
```

2. **Edit `.env` file:**

```bash
# backend/.env
NGROK_AUTH_TOKEN=your_ngrok_token_here
DEVICE=cuda  # or 'cpu' if no GPU
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
LORA_WEIGHTS=nikhilsoni700/dl_project_LoRA
```

3. **Configure ngrok:**

```bash
ngrok config add-authtoken YOUR_NGROK_TOKEN
```

### Frontend Configuration

The frontend will automatically detect the backend URL, but you can manually set it:

```bash
# frontend/.env
BACKEND_URL=http://localhost:5000
```

## Running the Application

### Option 1: Using Scripts (Recommended)

**Linux/macOS:**

Create `start_backend.sh`:
```bash
#!/bin/bash
cd backend
source ../venv/bin/activate
python app.py
```

Create `start_frontend.sh`:
```bash
#!/bin/bash
cd frontend
source ../venv/bin/activate
python web_app.py
```

Make executable:
```bash
chmod +x start_backend.sh start_frontend.sh
```

Run:
```bash
./start_backend.sh  # In terminal 1
./start_frontend.sh  # In terminal 2
```

**Windows:**

Create `start_backend.bat`:
```batch
@echo off
cd backend
call ..\venv\Scripts\activate
python app.py
```

Create `start_frontend.bat`:
```batch
@echo off
cd frontend
call ..\venv\Scripts\activate
python web_app.py
```

Run by double-clicking the batch files or in separate terminal windows.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
python app.py
```

Wait for:
```
INFO - Pipeline initialized successfully!
INFO - ngrok tunnel established at: https://xxxxx.ngrok-free.app
INFO - Starting Flask server on port 5000...
```

**Copy the ngrok URL** displayed in the terminal.

**Terminal 2 - Frontend:**
```bash
cd frontend
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
```

**Edit `web_app.py`** and update the BACKEND_URL if using ngrok:
```python
BACKEND_URL = "https://your-ngrok-url.ngrok-free.app"
```

Then start the frontend:
```bash
python web_app.py
```

Wait for:
```
INFO - Running on local URL:  http://127.0.0.1:7860
```

## Verification

### Step 1: Check Backend Health

Open a browser and go to:
```
http://localhost:5000/health
```

Or if using ngrok:
```
https://your-ngrok-url.ngrok-free.app/health
```

You should see:
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true
}
```

### Step 2: Access Frontend

Open your browser and navigate to:
```
http://127.0.0.1:7860
```

You should see the Gradio interface with:
- A text input for prompts
- Sliders for guidance scale and inference steps
- A generate button
- Example prompts

### Step 3: Generate Test Image

1. Enter a prompt: "A golden retriever playing in the snow"
2. Click "Generate Image"
3. Wait 30-60 seconds
4. Verify the image appears

## Troubleshooting

### Backend Issues

**Problem: "CUDA out of memory"**

Solutions:
```bash
# Reduce memory usage by adding to app.py:
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
```

Or use a smaller batch size/fewer inference steps.

**Problem: "Pipeline not loading"**

Solutions:
- Check internet connection (models are ~20GB)
- Clear cache: `rm -rf ~/.cache/huggingface/`
- Verify CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`

**Problem: "ngrok connection failed"**

Solutions:
- Verify auth token is correct
- Check ngrok status: `ngrok http 5000`
- The app will still work locally without ngrok

### Frontend Issues

**Problem: "Cannot connect to backend"**

Solutions:
- Verify backend is running
- Check the BACKEND_URL in web_app.py
- Test backend directly: `curl http://localhost:5000/health`

**Problem: "Request timeout"**

Solutions:
- Increase timeout in web_app.py: `TIMEOUT = 180`
- Check GPU utilization: `nvidia-smi`
- Reduce num_inference_steps

### General Issues

**Problem: "Module not found"**

Solutions:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Problem: "Permission denied"**

Solutions:
```bash
# Linux/macOS
chmod +x start_*.sh

# Or run with explicit python
python backend/app.py
```

**Problem: "Slow generation (5+ minutes)"**

Check:
- GPU is being used: Should see in backend logs "Using GPU: ..."
- Not running on CPU (will be very slow)
- Check GPU memory: `nvidia-smi`

## Performance Optimization

### Enable Memory Optimizations

Add to `backend/app.py` before initializing pipeline:

```python
# Enable memory efficient attention
pipeline.enable_attention_slicing()

# Enable VAE slicing
pipeline.enable_vae_slicing()

# For even better memory efficiency (slightly slower)
pipeline.enable_xformers_memory_efficient_attention()
```

### Use FP16

Already enabled by default for CUDA. If needed manually:

```python
pipeline = pipeline.to("cuda", torch_dtype=torch.float16)
```

## Next Steps

Now that your system is running:

1. **Try different prompts** - Experiment with the example prompts
2. **Adjust parameters** - Play with guidance scale and inference steps
3. **Train your own model** - See [TRAINING.md](docs/TRAINING.md)
4. **Customize the interface** - Modify `web_app.py` to add features
5. **Deploy** - Use ngrok or cloud services for public access

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review backend logs for errors
3. Open an issue on GitHub with:
   - Error message
   - System info (OS, GPU, Python version)
   - Steps to reproduce

## System Information Commands

Helpful commands for troubleshooting:

```bash
# Python version
python --version

# CUDA version
nvcc --version
nvidia-smi

# PyTorch + CUDA check
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Disk space
df -h  # Linux/macOS
dir    # Windows

# Memory
free -h  # Linux
vm_stat  # macOS
systeminfo  # Windows
```

---

**Congratulations!** Your Stable Diffusion XL + LoRA system is now set up and ready to use! 🎉