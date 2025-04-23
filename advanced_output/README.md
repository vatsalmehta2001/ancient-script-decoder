# Advanced Output Directory

This directory contains the trained model and associated files for the hieroglyph recognition system. The following files are included:

- `app_ready_model.h5`: The trained TensorFlow model used for hieroglyph recognition
- `class_mapping.json`: Mapping between class indices and Gardiner codes

## Model Details

The model is a Convolutional Neural Network (CNN) with the following specifications:
- Input shape: (224, 224, 3) - RGB images
- Output: 131 hieroglyph classes based on Gardiner notation
- Training accuracy: 82.66% (top-1), ~95% (top-3)

## Limitations

As noted in the main README, the current model has limitations when used on real-world images. The integration between detection and recognition systems may produce inconsistent results, especially on complex scenes with multiple hieroglyphs. 