# Ancient Egyptian Script Decipherment Assistant

My passion for ancient Egyptian history was ignited through conversations on Joe Rogan's podcast featuring Graham Hancock and Randall Carlson. Their discussions led me down a fascinating rabbit hole exploring Egyptian mythology, hieroglyphics, and the mysteries surrounding their ancient civilization. I was particularly captivated to learn that even Nikola Tesla was once extremely passionate about Egyptian origins and the true purpose of the pyramids. This project represents my first step in combining my interests in ancient history with modern AI technology.

[![Hieroglyph Recognition Demo](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/demo_screenshot.png)](https://hieroglyph-analyzer.streamlit.app/)

## ⚠️ Important Note on Model Accuracy

While the model reports a training accuracy of 82.66% on the training dataset, there are significant limitations when used on new images in the wild. The current integration between the detection system and recognition model often produces inconsistent or incorrect predictions. This discrepancy occurs because:

1. The model was trained on isolated, clean hieroglyph images but is being applied to complex scenes with multiple glyphs
2. The sliding window detection approach is sensitive to scale, rotation, and image quality
3. Real-world hieroglyphs often appear in contexts very different from the training data

**This is an experimental project** and the current implementation should be considered a proof-of-concept rather than a production-ready system. Future iterations will aim to improve the accuracy and robustness of both detection and classification.

## Project Overview

This project uses deep learning to recognize and decipher ancient Egyptian hieroglyphs. The system can identify individual hieroglyphs from the Gardiner Sign List using computer vision and machine learning techniques.

The application utilizes a Streamlit interface for ease of use and deployment.

**Key Features:**
- Advanced hieroglyph detection and recognition with 82.66% top-1 accuracy on the training dataset
- Interactive web interface for uploading and analyzing images
- Detailed information about detected hieroglyphs and their meanings
- Visual feedback with bounding boxes and confidence scores

## How to Use

### Online Version

Visit the live demo at [Streamlit Cloud](https://hieroglyph-analyzer.streamlit.app/)

### Local Installation

1. Clone this repository:
```bash
git clone https://github.com/vatsalmehta2001/ancient-script-decoder.git
cd ancient-script-decoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## System Architecture

The system employs a multi-stage process:

1. **Detection**: A sliding window approach scans the image at multiple scales to locate potential hieroglyphs.
2. **Recognition**: Each detected region is analyzed by a deep learning model.
3. **Classification**: The model predicts the Gardiner code for each detected hieroglyph, along with a confidence score.
4. **Visualization**: The results are visualized with bounding boxes and labels on the original image.

## Model Architecture

The system uses a custom-designed hierarchical neural network architecture that combines:

1. **Convolutional Neural Network (CNN)** with residual connections and squeeze-excitation blocks
2. **EfficientNetV2S** for transfer learning capabilities
3. **Vision Transformer (ViT)** for attention-based feature extraction

![Model Architecture Diagram](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/model_architecture_diagram.png)
*Detailed architecture diagram of the advanced 82.66% accuracy model showing the three parallel branches (CNN, EfficientNet, and Vision Transformer) and how they're combined for hieroglyph classification.*

### Advanced Training Techniques

The model was trained using several advanced techniques:

- Learning rate warmup with cosine decay
- Label smoothing for better generalization
- Mixed precision training for improved performance
- Dropout and batch normalization for regularization

## Model Performance Visualizations

### Sample Hieroglyphs
![Sample Hieroglyphs](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/sample_hieroglyphs.png)
*A selection of hieroglyph images from the advanced model training dataset (82.66% accuracy model) showing the variety of symbols and their Gardiner codes.*

### Data Augmentation
![Augmentation Examples](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/augmentation_examples.png)
*Examples of data augmentation techniques used in the advanced model training (82.66% accuracy) applied to hieroglyph images, increasing the diversity of the training data.*

### Class Distribution
![Class Distribution](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/class_distribution.png)
*Distribution of the top 30 hieroglyph classes by sample count in the advanced model dataset (82.66% accuracy model).*

### Performance Metrics

- **Top-1 Accuracy**: 82.66% (on training and validation data)
- **Top-3 Accuracy**: ~95% (on training and validation data)
- **Real-world Performance**: Significantly lower than the reported metrics - expect inconsistent results on new images

## Dataset

This project uses a dataset of hieroglyph images organized by Gardiner code classification. The dataset contains high-quality images representing distinct hieroglyphic symbols in various contexts.

## Technical Implementation

### TensorFlow-Optimized Learning Rate Scheduler
The model uses a custom implementation of cosine decay with warmup that is fully TensorFlow-compatible, avoiding Python-based conditional logic that would cause graph execution errors. This scheduler:

- Implements a smooth transition from warmup to decay phase
- Uses `tf.cond` for flow control instead of Python conditionals
- Ensures gradients flow properly through the computation graph
- Facilitates better model convergence

### Advantages of the Advanced Model
- **Robust Feature Extraction**: The architectural design allows for multi-scale feature extraction
- **Efficient Training**: Mixed-precision and optimized learning rate scheduling reduce training time
- **Better Generalization**: Advanced regularization techniques prevent overfitting
- **Production-Ready**: Model is saved in multiple formats for easy deployment

## Project Components

- `streamlit_app.py`: Main Streamlit application for web interface
- `hieroglyph_analyzer.py`: Core class for hieroglyph detection and recognition
- `test_hieroglyph_analyzer.py`: Test script for the analyzer
- `advanced_output/`: Model files and class mappings
- `test_images/`: Sample images for testing the system

## Acknowledgments

- Dataset compilation inspired by the [EgyptianHieroglyphicText](https://github.com/rfuentesfe/EgyptianHieroglyphicText) repository
- Gardiner Sign List for hieroglyph categorization
- Inspiration: Graham Hancock, Randall Carlson, and others exploring ancient civilizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Work

- Improved model training with more diverse and representative datasets
- Better hieroglyph detection using modern object detection techniques (YOLO, SSD)
- Multi-hieroglyph sequence translation for complete text interpretation
- Integration with historical context database for deeper meaning extraction
- Mobile application for on-site archaeological use
- Expansion to other ancient writing systems
