# Ancient Egyptian Script Decipherment Assistant

My passion for ancient Egyptian history was ignited through conversations on Joe Rogan's podcast featuring Graham Hancock and Randall Carlson. Their discussions led me down a fascinating rabbit hole exploring Egyptian mythology, hieroglyphics, and the mysteries surrounding their ancient civilization. I was particularly captivated to learn that even Nikola Tesla was once extremely passionate about Egyptian origins and the true purpose of the pyramids. This project represents my first step in combining my interests in ancient history with modern AI technology.

![Hieroglyph Recognition Demo](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/demo_screenshot.png)

## Project Overview

This project uses deep learning to recognize and decipher ancient Egyptian hieroglyphs. The system can identify individual hieroglyphs from the Gardiner Sign List using computer vision and machine learning techniques.

**Note: This project is continuously evolving.** Currently, the system can identify hieroglyphic symbols with high confidence levels using a state-of-the-art advanced model I built from scratch during my university studies.

## Features

- **Advanced Hieroglyph Recognition**: Identifies individual hieroglyphs from images with high accuracy (82.66% top-1, 95% top-3 accuracy)
- **State-of-the-Art Architecture**: Implements TensorFlow-based model that combines:
  1. **Convolutional Neural Network (CNN)** with residual connections and squeeze-excitation blocks
  2. **EfficientNetV2S** for transfer learning capabilities 
  3. **Vision Transformer (ViT)** for attention-based feature extraction
- **Optimized Learning Rate Scheduling**: Uses cosine decay with warmup for improved convergence
- **Web Interface**: User-friendly Streamlit application for uploading and analyzing hieroglyphic texts
- **Comprehensive Evaluation**: Detailed metrics and visualizations to understand model performance
- **Detection System**: Automatically detects and segments individual hieroglyphs in complex scenes

![Model Architecture](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/model_architecture.png)

## Dataset

This project uses a dataset compiled from various sources containing:

- High-quality hieroglyph images organized by Gardiner code classification
- Multiple classes representing distinct hieroglyphic symbols
- Various lighting, angles, and contexts for robust recognition

The raw dataset was obtained from the [EgyptianHieroglyphicText](https://github.com/rfuentesfe/EgyptianHieroglyphicText/tree/main) repository, which provides a comprehensive collection of hieroglyph images structured according to Gardiner's classification system. This dataset consists of 310 classes and 13,729 images representing hieroglyphs on different materials, including carved or painted stone stelae.

## Advanced Training Techniques

I implemented several advanced techniques to achieve high recognition accuracy:

- Learning rate warmup with cosine decay
- Label smoothing for better generalization
- Mixed precision training for improved performance
- Dropout and batch normalization for regularization

![Example Recognition](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/example_recognition.png)

## Project Structure

```
ancient-script-decoder/
├── dataset_new/                  # Improved hierarchical dataset for training (not included in repo)
├── advanced_output/              # Model output, checkpoints, and evaluation metrics
│   └── app_ready_model.h5        # Ready-to-use model for the app
├── app.py                        # Streamlit web application
├── advanced_model.py             # Advanced model architecture implementation
├── advanced_preprocessing.py     # Advanced data preprocessing utilities
├── train_advanced.py             # Advanced model training script
├── run_advanced_training.sh      # Convenience script for training the model
├── setup_environment.sh          # Script to set up the environment
├── export_model_for_app.py       # Utility to export model for use in the app
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vatsalmehta2001/ancient-script-decoder.git
cd ancient-script-decoder
```

2. Run the setup script to create necessary directories and install dependencies:
```bash
./setup_environment.sh
```

3. Download the dataset (only needed for training new models):
   - Visit the [EgyptianHieroglyphicText repository](https://github.com/rfuentesfe/EgyptianHieroglyphicText/tree/main)
   - Clone the repository and copy the hieroglyph images into the `dataset_new/` directory
   - Ensure images are organized by Gardiner code (e.g., a1, a2, etc.)

## Usage

### Running the Application

The trained model (app_ready_model.h5) is included in the repository, so you can run the application immediately:

```bash
streamlit run app.py
```

The application will open in your web browser, allowing you to:
1. Upload images containing hieroglyphs
2. View real-time predictions with confidence scores
3. Explore the top predictions for each symbol

### Advanced Model Training

To train the advanced model with optimal parameters:

```bash
./run_advanced_training.sh
```

This script configures and runs the training process with optimized hyperparameters, producing a high-performance model.

### Manual Advanced Training

For custom training configurations:

```bash
python train_advanced.py --model_type advanced_cnn --epochs 200 --batch_size 32 --patience 30 --img_height 224 --img_width 224
```

### Preparing Model for App

To prepare a trained model for use in the app:

```bash
python export_model_for_app.py --checkpoint advanced_output/advanced_model_TIMESTAMP/model_checkpoint_XX_ACCURACY.h5 --output advanced_output/app_ready_model.h5
```

## Advanced Model Architecture

My state-of-the-art model employs several advanced techniques:

### Advanced CNN Architecture
- Deep convolutional neural network with residual connections
- Batch normalization and dropout for regularization
- Swish activation functions for improved gradient flow
- Squeeze-and-excite blocks for channel-wise attention

### Learning Rate Optimization
- TensorFlow-friendly cosine decay scheduler with warmup
- Gradual learning rate warmup (5% of epochs) followed by cosine decay
- Adaptive learning rate adjustments based on validation performance

### Advanced Training Features
- Mixed-precision training for improved performance
- Early stopping mechanism to prevent overfitting
- Model checkpointing to save the best-performing model versions
- Comprehensive TensorBoard logging for monitoring training progress

## Development Process

I developed this project over several months as part of my university studies, following this process:

1. **Research**: Studied hieroglyphic symbols and existing classification methods
2. **Data Collection**: Gathered and curated hieroglyph images from various sources
3. **Model Development**: Experimented with different architectures, hyperparameters
4. **Training Infrastructure**: Built robust training pipeline with checkpointing and metrics
5. **Application Development**: Created a user-friendly interface for real-world use
6. **Optimization**: Fine-tuned the model for better accuracy and performance

## Model Performance and Checkpoints

The advanced model training reached epoch 68 before early stopping was triggered due to no further improvement in validation accuracy. The total training time was approximately 3.5 hours on a standard GPU-equipped machine. Key performance metrics:

- **Best Validation Accuracy**: 82.66% (achieved at epoch 60)
- **Top-3 Accuracy**: ~95% (consistently throughout later epochs)
- **Training Accuracy**: ~85-87% (at later stages of training)

### Checkpoint Evolution

The model showed clear progression through training:

1. **Initial Learning Phase (Epochs 1-10)**:
   - Accuracy improved from 3.42% to 26.77%
   - Model began to recognize basic patterns

2. **Rapid Improvement Phase (Epochs 11-30)**:
   - Accuracy jumped from 37.41% to 71.63%
   - Model established foundational feature recognition capabilities
   - Learning rate adjustments helped maintain steady improvements

3. **Refinement Phase (Epochs 31-50)**:
   - Accuracy improved more gradually from 71.63% to 79.70%
   - Model began focusing on more difficult distinctions between similar hieroglyphs
   - Top-3 accuracy reached ~92%

4. **Fine-tuning Phase (Epochs 51-68)**:
   - Accuracy reached its peak at 82.66% (epoch 60)
   - Validation accuracy plateaued, triggering early stopping at epoch 68
   - Top-3 accuracy stabilized at ~95%

## Visualizations

The training process generated several informative visualizations that help understand both the dataset and the model's performance:

### Sample Hieroglyphs
![Sample Hieroglyphs](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/sample_hieroglyphs.png)
*A selection of hieroglyph images from the training dataset showing the variety of symbols and their Gardiner codes.*

### Data Augmentation
![Augmentation Examples](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/augmentation_examples.png)
*Examples of data augmentation techniques applied to hieroglyph images, increasing the diversity of the training data.*

### Class Distribution
![Class Distribution](https://github.com/vatsalmehta2001/ancient-script-decoder/raw/main/images/class_distribution.png)
*Distribution of the top 30 hieroglyph classes by sample count in the dataset.*

## Technical Details

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

## Future Improvements

Planned improvements include:

- Expanding the hieroglyph database with more context-specific examples
- Implementing ensemble techniques to further improve accuracy
- Adding context-aware translation by understanding hieroglyph sequences
- Developing a system that can interpret complete hieroglyphic texts
- Building a mobile application for on-site archaeological use

## Creating a More Complete Documentation

To complete the documentation with proper images, I need to:

1. Create the `images/` directory
2. Take screenshots of the application in action
3. Create diagrams of the model architecture
4. Add examples of successful recognitions
5. Replace the image placeholder URLs with actual screenshots

## Credits

- Dataset compilation and organization from various academic sources
- Gardiner Sign List for hieroglyph categorization
- Inspiration: Graham Hancock, Randall Carlson, and others exploring ancient civilizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or dataset in your research, please cite both the original dataset work and the enhanced dataset:

```
Franken, M., & van Gemert, J. C. (2013). Automatic Egyptian Hieroglyph Recognition by Retrieving Images as Texts. 
In Proceedings of the 21st ACM International Conference on Multimedia (pp. 765-768). ACM.

R. Fuentes-Ferrer, J. Duque-Domingo, and P.J. Herrera (2025). Recognition of Egyptian Hieroglyphic Texts through 
Focused Generic Segmentation and Cross-Validation Voting. Applied Soft Computing, 
DOI: https://doi.org/10.1016/j.asoc.2025.112793
```
