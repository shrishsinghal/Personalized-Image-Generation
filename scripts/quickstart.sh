#!/bin/bash

# Quick Start Script for SDXL + LoRA Project
# This script automates the setup process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Start
clear
print_header "SDXL + LoRA Quick Start Setup"

# Step 1: Check prerequisites
print_header "Step 1: Checking Prerequisites"

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
    print_info "Python version: $PYTHON_VERSION"
else
    print_error "Python 3.8+ is required. Please install Python first."
    exit 1
fi

# Check CUDA
if check_command nvidia-smi; then
    print_info "GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    print_error "NVIDIA GPU not detected. Training will be slow on CPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check git
check_command git || {
    print_error "Git is required. Please install git first."
    exit 1
}

# Step 2: Create virtual environment
print_header "Step 2: Setting Up Virtual Environment"

if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
else
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
print_success "Pip upgraded"

# Step 3: Install dependencies
print_header "Step 3: Installing Dependencies"

# Ask which components to install
echo "Which components do you want to install?"
echo "1) Backend only"
echo "2) Frontend only"
echo "3) Training only"
echo "4) All components (recommended)"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        print_info "Installing backend dependencies..."
        pip install -r backend/requirements.txt
        print_success "Backend dependencies installed"
        ;;
    2)
        print_info "Installing frontend dependencies..."
        pip install -r frontend/requirements.txt
        print_success "Frontend dependencies installed"
        ;;
    3)
        print_info "Installing training dependencies..."
        pip install -r training/requirements.txt
        print_success "Training dependencies installed"
        ;;
    4)
        print_info "Installing all dependencies (this may take 10-30 minutes)..."
        pip install -r backend/requirements.txt
        pip install -r frontend/requirements.txt
        pip install -r training/requirements.txt
        print_success "All dependencies installed"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Step 4: Configure environment
print_header "Step 4: Configuration"

# Backend configuration
if [ "$choice" == "1" ] || [ "$choice" == "4" ]; then
    if [ ! -f "backend/.env" ]; then
        print_info "Setting up backend configuration..."
        
        read -p "Enter your ngrok auth token (or press Enter to skip): " NGROK_TOKEN
        
        cat > backend/.env << EOF
# Backend Configuration
NGROK_AUTH_TOKEN=${NGROK_TOKEN:-your_token_here}
DEVICE=cuda
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
LORA_WEIGHTS=nikhilsoni700/dl_project_LoRA
EOF
        print_success "Backend configuration created"
    else
        print_info "Backend configuration already exists"
    fi
fi

# Frontend configuration
if [ "$choice" == "2" ] || [ "$choice" == "4" ]; then
    if [ ! -f "frontend/.env" ]; then
        print_info "Setting up frontend configuration..."
        
        cat > frontend/.env << EOF
# Frontend Configuration
BACKEND_URL=http://localhost:5000
EOF
        print_success "Frontend configuration created"
    else
        print_info "Frontend configuration already exists"
    fi
fi

# Step 5: Download training script
if [ "$choice" == "3" ] || [ "$choice" == "4" ]; then
    print_header "Step 5: Downloading Training Script"
    
    if [ ! -f "training/train_dreambooth_lora_sdxl.py" ]; then
        print_info "Downloading DreamBooth training script..."
        wget -q -P training/ https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
        print_success "Training script downloaded"
    else
        print_info "Training script already exists"
    fi
fi

# Step 6: Create helper scripts
print_header "Step 6: Creating Helper Scripts"

# Backend start script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source ../venv/bin/activate
python app.py
EOF
chmod +x start_backend.sh
print_success "Created start_backend.sh"

# Frontend start script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
source ../venv/bin/activate
python web_app.py
EOF
chmod +x start_frontend.sh
print_success "Created start_frontend.sh"

# Quick test script
cat > test_setup.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
print('\nSetup test passed!')
"
EOF
chmod +x test_setup.sh
print_success "Created test_setup.sh"

# Step 7: Test installation
print_header "Step 7: Testing Installation"

print_info "Running setup test..."
./test_setup.sh

# Final instructions
print_header "Setup Complete!"

echo "Your SDXL + LoRA environment is ready!"
echo ""
echo "Next steps:"
echo ""

if [ "$choice" == "1" ] || [ "$choice" == "4" ]; then
    echo "To start the backend:"
    echo "  ./start_backend.sh"
    echo ""
fi

if [ "$choice" == "2" ] || [ "$choice" == "4" ]; then
    echo "To start the frontend:"
    echo "  ./start_frontend.sh"
    echo ""
fi

if [ "$choice" == "3" ] || [ "$choice" == "4" ]; then
    echo "To train a model:"
    echo "  python training/train_lora.py --help"
    echo ""
fi

echo "For more information:"
echo "  - Setup guide: SETUP.md"
echo "  - Training guide: docs/TRAINING.md"
echo "  - Examples: docs/EXAMPLES.md"
echo "  - API docs: docs/API.md"
echo ""

print_success "Happy generating! 🎨"