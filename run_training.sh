#!/bin/bash

# ==============================================================================
# YOLOv10-FEM One-Click Training Script
# This script starts the training process for the Frequency-Enhanced Mechanism.
# ==============================================================================

echo "==========================================================="
echo "🚀 Starting YOLOv10-FEM Training..."
echo "==========================================================="

# Ensure we are in the correct directory
cd "F:/nus/icml/yolov10" || { echo "❌ Failed to change directory. Please check the path."; exit 1; }

# Check if Python is available
if ! command -v python &> /dev/null
then
    echo "❌ Python could not be found. Please ensure your Conda environment is activated."
    exit 1
fi

echo "✅ Environment ready. Launching pure Python training script..."
echo "   - Dataset: F:/nus/icml/split_5"
echo "   - Modules: FDAF (Frequency-Domain Attention Fusion)"
echo "   - Loss: L_freq (Frequency-Consistency Loss)"
echo "-----------------------------------------------------------"

# Run the training script
python train_frequency.py

echo "==========================================================="
echo "🎉 Training script execution finished."
echo "   Check the 'runs/detect/' folder for your results and weights."
echo "==========================================================="
