#!/bin/bash

# RECONTEXT Model Downloader
# Downloads required pretrained models for the RECONTEXT system
# Author: Michael Chen
# Date: 2024-01-15

# Set up constants
MODELS_DIR="$HOME/.recontext/models"
TEMP_DIR="/tmp/recontext_downloads"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"

# Make sure directories exist
mkdir -p "$MODELS_DIR"
mkdir -p "$TEMP_DIR"

# Function to download a model if it doesn't exist
download_model() {
    local model_name=$1
    local model_url=$2
    local model_path="$MODELS_DIR/$model_name"
    
    if [ -f "$model_path" ]; then
        echo "✓ $model_name already exists, skipping download"
    else
        echo "⬇️ Downloading $model_name..."
        
        # Try curl first, fallback to wget
        if command -v curl &> /dev/null; then
            curl -L "$model_url" -o "$model_path.tmp"
        elif command -v wget &> /dev/null; then
            wget "$model_url" -O "$model_path.tmp"
        else
            echo "❌ Error: Neither curl nor wget is available. Please install one of them."
            return 1
        fi
        
        # Check if download was successful
        if [ $? -eq 0 ]; then
            mv "$model_path.tmp" "$model_path"
            echo "✓ Downloaded $model_name successfully"
        else
            echo "❌ Failed to download $model_name"
            rm -f "$model_path.tmp"
            return 1
        fi
    fi
}

# Function to check Python dependencies
check_python_deps() {
    # Try to import required packages
    python -c "import torch; import open3d; import cv2" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "❌ Missing required Python dependencies."
        echo "Please run: pip install -r $PROJ_ROOT/requirements.txt"
        return 1
    fi
    
    return 0
}

# Print welcome message
echo "================================================================"
echo "RECONTEXT Model Downloader"
echo "This script will download the pretrained models needed for RECONTEXT"
echo "Models will be stored in: $MODELS_DIR"
echo "================================================================"

# Check Python dependencies
check_python_deps
if [ $? -ne 0 ]; then
    exit 1
fi

# Define models to download
declare -A MODELS=(
    ["mask2former_coco.pkl"]="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl"
    ["swin_large_patch4_window12_384_22k.pth"]="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"
    ["clip_ViT-B-32.pt"]="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    ["scene_query_t5_base.bin"]="https://huggingface.co/t5-base/resolve/main/pytorch_model.bin"
)

# Count number of models
total_models=${#MODELS[@]}
current=1

# Download each model
for model_name in "${!MODELS[@]}"; do
    echo "[$current/$total_models] Processing $model_name..."
    download_model "$model_name" "${MODELS[$model_name]}"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error downloading $model_name. Exiting."
        exit 1
    fi
    
    ((current++))
done

# Additional operations for specific models
echo "Setting up model configurations..."

# Generate model configs
python "$PROJ_ROOT/scripts/process_dataset.py" --setup-models

echo "================================================================"
echo "✅ All models downloaded successfully!"
echo "You can now run RECONTEXT with full functionality."
echo "================================================================"
 
 The script is quite simple and self-explanatory. It downloads the required models and sets up the model configurations. 
 Step 3: Run the Model Downloader 
 Now, you can run the model downloader script to download the required models. 
 bash scripts/download_models.sh
 
 The script will download the models and set up the model configurations. 
 Step 4: Run the RECONTEXT System 
 Finally, you can run the RECONTEXT system with the following command: 
 python recontext.py
 
 The system will start and you can interact with it using the command-line interface. 
 Conclusion 
 In this tutorial, you learned how to build a context-aware image retrieval system using the RECONTEXT system. You learned how to set up the system, download the required models, and run the system. You can further extend the system by adding more models or customizing the system to suit your needs. 
 If you want to learn more about the RECONTEXT system, you can refer to the  official documentation. 
 Happy coding! 
 Peer Review Contributions by:  Lalithnarayan C