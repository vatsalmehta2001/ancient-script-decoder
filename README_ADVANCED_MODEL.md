# Advanced Hieroglyph Recognition Model

This project implements a state-of-the-art hieroglyph recognition system using advanced deep learning techniques. The model is designed to achieve significantly higher accuracy than previous approaches by leveraging ensemble architecture, advanced augmentation, and modern training techniques.

## Features

- **Advanced Model Architecture**: Combines CNN, EfficientNetV2S, and Vision Transformer in an ensemble for superior recognition performance
- **Enhanced Preprocessing**: Sophisticated image preprocessing pipeline with advanced augmentation techniques
- **Improved Training Process**: Using mixed-precision training, learning rate scheduling, and regularization techniques
- **Comprehensive Evaluation**: Detailed metrics including confusion matrices and per-class performance analysis
- **Production-Ready Export**: Models exported in multiple formats for deployment

## Requirements

All requirements are listed in `requirements_advanced.txt`. Install them with:

```bash
pip install -r requirements_advanced.txt
```

## Dataset

The model is designed to work with the new hieroglyph dataset structure in the `dataset_new` directory. This dataset contains multiple hieroglyph classes organized in a directory structure where each subdirectory is named after the Gardiner code for that hieroglyph class.

## Quick Start

To train the advanced model with optimal parameters, simply run:

```bash
./run_advanced_training.sh
```

This script will:
1. Install required packages
2. Create necessary output directories
3. Run the training process with optimized hyperparameters
4. Save the model and evaluation results

## Manual Training

If you want to customize the training process, you can run the training script directly:

```bash
python train_advanced.py --dataset_path dataset_new --output_dir custom_output --model_type ensemble
```

### Important Parameters

- `--dataset_path`: Path to the dataset directory (default: `dataset_new`)
- `--output_dir`: Directory to save output files (default: `hieroglyph_recognition_advanced`)
- `--img_height` and `--img_width`: Image dimensions (default: 128×128)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Maximum number of training epochs (default: 100)
- `--patience`: Early stopping patience (default: 15)
- `--model_type`: Model architecture, choices: `ensemble` or `advanced_cnn` (default: `ensemble`)
- `--disable_mixed_precision`: Disable mixed precision training for better compatibility
- `--visualize_model`: Generate model architecture visualization
- `--export_model`: Export the model for deployment

## Model Architecture

The advanced model uses an ensemble approach combining three powerful architectures:

1. **Advanced CNN Branch**: A sophisticated CNN with residual connections and modern activation functions
2. **EfficientNetV2S Branch**: Transfer learning from a pre-trained EfficientNetV2S model
3. **Vision Transformer Branch**: A custom ViT implementation for capturing global relationships

These three branches are combined with a powerful classification head to produce final predictions.

## Performance

The advanced model is expected to achieve significantly higher accuracy than previous approaches:

- Better handling of difficult hieroglyph classes
- Improved robustness to image variations and noise
- Higher top-1 and top-5 accuracy
- Better generalization to unseen examples

## Using the Trained Model

After training, the model will be saved in the output directory. You can use it for prediction with:

```python
from advanced_model import AdvancedHieroglyphModel
from advanced_preprocessing import AdvancedHieroglyphProcessor
import json

# Load class mapping
with open('advanced_output/class_mapping.json', 'r') as f:
    mapping_data = json.load(f)

# Initialize processor and model
processor = AdvancedHieroglyphProcessor(img_size=(224, 224))
processor.inv_class_mapping = mapping_data['inv_class_mapping']

# Load model
model = AdvancedHieroglyphModel(num_classes=len(mapping_data['inv_class_mapping']), img_size=(224, 224))
model.model = tf.keras.models.load_model('advanced_output/advanced_model_TIMESTAMP/final_model.h5')

# Make predictions
result = model.predict_with_confidence('path/to/image.jpg', processor)
print(f"Predicted class: {result['top_prediction']['gardiner_code']}")
print(f"Confidence: {result['top_prediction']['confidence']:.4f}")
```

## Export Formats

The model can be exported in multiple formats for deployment:

- **TensorFlow SavedModel**: For TensorFlow Serving or direct loading
- **TensorFlow Lite**: For mobile and edge devices
- **H5 Format**: For easy loading in Keras

## Acknowledgments

This project builds upon the initial hieroglyph recognition system, significantly improving its performance with state-of-the-art deep learning techniques. 