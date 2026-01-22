#!/bin/bash
# Setup script for track_and_crop with SAM2
# This script creates a conda environment and installs all requirements

set -e  # Exit on error

ENV_NAME="${1:-track_and_crop}"
PYTHON_VERSION="3.11"

echo "============================================"
echo "Setting up track_and_crop environment"
echo "Environment name: $ENV_NAME"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create conda environment
echo ""
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
# Using PyTorch 2.5.1 with CUDA 12.4 (recommended for SAM2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Clone and install SAM2 into thirdparty directory
echo ""
echo "Setting up thirdparty directory..."
mkdir -p "$SCRIPT_DIR/thirdparty"

echo "Cloning SAM2 repository..."
if [ -d "$SCRIPT_DIR/thirdparty/sam2" ]; then
    echo "SAM2 directory already exists, pulling latest changes..."
    cd "$SCRIPT_DIR/thirdparty/sam2"
    git pull
else
    cd "$SCRIPT_DIR/thirdparty"
    git clone https://github.com/facebookresearch/sam2.git
    cd "$SCRIPT_DIR/thirdparty/sam2"
fi

echo ""
echo "Installing SAM2 dependencies..."
pip install -e .

# Return to script directory
cd "$SCRIPT_DIR"

# Install additional requirements
echo ""
echo "Installing additional requirements..."
pip install -r requirements.txt

# Download SAM2 checkpoints
echo ""
echo "Setting up checkpoints directory..."
mkdir -p "$SCRIPT_DIR/checkpoints"

# Create checkpoint download script if it doesn't exist
if [ ! -f "$SCRIPT_DIR/checkpoints/download_ckpts.sh" ]; then
    cat > "$SCRIPT_DIR/checkpoints/download_ckpts.sh" << 'EOF'
#!/bin/bash
# Download SAM2.1 checkpoints

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# SAM2.1 checkpoints
declare -A CHECKPOINTS=(
    ["sam2.1_hiera_tiny.pt"]="sam2.1_hiera_tiny.pt"
    ["sam2.1_hiera_small.pt"]="sam2.1_hiera_small.pt"
    ["sam2.1_hiera_base_plus.pt"]="sam2.1_hiera_base_plus.pt"
    ["sam2.1_hiera_large.pt"]="sam2.1_hiera_large.pt"
)

echo "Downloading SAM2.1 checkpoints..."

for ckpt in "${!CHECKPOINTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "Downloading $ckpt..."
        wget -q --show-progress "${BASE_URL}/${CHECKPOINTS[$ckpt]}" -O "$ckpt"
    else
        echo "$ckpt already exists, skipping."
    fi
done

echo "Done!"
EOF
    chmod +x "$SCRIPT_DIR/checkpoints/download_ckpts.sh"
fi

# Ask user which checkpoint to download
echo ""
echo "Which SAM2.1 checkpoint would you like to download?"
echo "  1) tiny     (38.9M params, fastest)"
echo "  2) small    (46M params)"
echo "  3) base_plus (80.8M params)"
echo "  4) large    (224.4M params, best quality)"
echo "  5) all      (download all checkpoints)"
echo "  6) skip     (download later manually)"
read -p "Enter choice [1-6, default=4]: " choice

cd "$SCRIPT_DIR/checkpoints"

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

download_checkpoint() {
    local name=$1
    if [ ! -f "$name" ]; then
        echo "Downloading $name..."
        wget -q --show-progress "${BASE_URL}/${name}" -O "$name"
    else
        echo "$name already exists."
    fi
}

case $choice in
    1) download_checkpoint "sam2.1_hiera_tiny.pt" ;;
    2) download_checkpoint "sam2.1_hiera_small.pt" ;;
    3) download_checkpoint "sam2.1_hiera_base_plus.pt" ;;
    4|"") download_checkpoint "sam2.1_hiera_large.pt" ;;
    5)
        download_checkpoint "sam2.1_hiera_tiny.pt"
        download_checkpoint "sam2.1_hiera_small.pt"
        download_checkpoint "sam2.1_hiera_base_plus.pt"
        download_checkpoint "sam2.1_hiera_large.pt"
        ;;
    6) echo "Skipping checkpoint download. Run 'cd checkpoints && ./download_ckpts.sh' later." ;;
    *) echo "Invalid choice, skipping download." ;;
esac

cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Example usage:"
echo "  python track_and_crop.py \\"
echo "    --video input.mp4 \\"
echo "    --bbox '100,150,300,400' \\"
echo "    --output output.mp4"
echo ""
echo "For more options, run:"
echo "  python track_and_crop.py --help"
echo "============================================"
