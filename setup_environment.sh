#!/bin/bash

# Setup environment for Ancient Egyptian Script Decipherment Assistant
# This script installs dependencies and prepares the environment for Streamlit deployment

echo "Setting up environment for Ancient Egyptian Script Decipherment Assistant..."

# Create necessary directories
mkdir -p uploaded_images
mkdir -p analyzer_output
mkdir -p test_images

# Check if python3 is installed
if command -v python3 &>/dev/null; then
    echo "Python 3 is installed"
else
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

# Check if pip is installed
if command -v pip3 &>/dev/null; then
    echo "pip is installed"
else
    echo "pip is not installed. Please install pip before continuing."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Check if tensorflow is installed correctly
echo "Verifying TensorFlow installation..."
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Check if model file exists
if [ -f "./advanced_output/app_ready_model.h5" ]; then
    echo "Model file found."
else
    echo "Warning: Model file not found at ./advanced_output/app_ready_model.h5"
    echo "Please ensure the model file is available before running the application."
fi

# Check if class mapping file exists
if [ -f "./advanced_output/class_mapping.json" ]; then
    echo "Class mapping file found."
else
    echo "Warning: Class mapping file not found at ./advanced_output/class_mapping.json"
    echo "Please ensure the class mapping file is available before running the application."
fi

echo "Setup complete!"
echo "You can now run the application with: streamlit run streamlit_app.py" 