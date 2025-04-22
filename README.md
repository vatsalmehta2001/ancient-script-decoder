# Ancient Egyptian Script Decipherment Assistant

My passion for ancient Egyptian history was ignited through conversations on Joe Rogan's podcast featuring Graham Hancock and Randall Carlson. Their discussions led me down a fascinating rabbit hole exploring Egyptian mythology, hieroglyphics, and the mysteries surrounding their ancient civilization. I was particularly captivated to learn that even Nikola Tesla was once extremely passionate about Egyptian origins and the true purpose of the pyramids. This project represents my first step in combining my interests in ancient history with modern AI technology.

## Project Overview
This project uses deep learning to recognize and decipher ancient Egyptian hieroglyphs. The system can identify individual hieroglyphs from the Gardiner Sign List using computer vision and machine learning techniques.

**Note: This project is just the beginning and is nowhere near finished.** Currently, the system can identify hieroglyphic symbols with confidence levels, but a comprehensive translation system with complete meanings for each code is still in development.

## Features

- **Hieroglyph Recognition**: Identifies individual hieroglyphs from images with high accuracy (93.8% on test set)
- **Multiple Model Architectures**: Includes CNN, EfficientNet, and Transformer-based models
- **Web Interface**: User-friendly Streamlit application for uploading and analyzing hieroglyphic texts
- **Comprehensive Evaluation**: Detailed metrics and visualizations to understand model performance
- **Detection System**: Automatically detects and segments individual hieroglyphs in complex scenes

## Dataset

This project uses a dataset compiled by Morris Franken, complementary to the paper titled "Automatic Egyptian Hieroglyph Recognition by Retrieving Images as Texts" (ACM Conference on Multimedia, 2013). The dataset contains:

- 4,210 hieroglyph images (of which 179 are labeled as UNKNOWN)
- 171 distinct classes (according to the Gardiner Sign List)
- Images extracted from 10 different plates from "The Pyramid of Unas" (Alexandre Piankoff, 1955)
- Both manually annotated and automatically detected hieroglyphs

## Project Structure

```
ancient-script-decoder/
├── Dataset/                      # Original dataset folder
├── app.py                        # Streamlit web application
├── evaluate.py                   # Evaluation script
├── model.py                      # Model architecture implementations
├── preprocessing.py              # Data preprocessing utilities
├── train.py                      # Training script
├── hieroglyph_recognition/       # Generated folder with processed data and models
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ancient-script-decoder.git
cd ancient-script-decoder
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing and Model Training

To preprocess the data and train a model:

```bash
python train.py --model_type cnn --epochs 50 --batch_size 32 --export_model
```

Available model types:
- `cnn`: Standard convolutional neural network
- `efficientnet`: Transfer learning with EfficientNetB0
- `transformer`: Vision Transformer-inspired architecture

### Model Evaluation

To evaluate a trained model:

```bash
python evaluate.py
```

This will generate detailed evaluation metrics, confusion matrices, and visualizations in the `hieroglyph_recognition/evaluation` directory.

### Web Interface

To run the web interface:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Model Architectures

### CNN Model

A standard convolutional neural network with the following architecture:
- Three convolutional blocks with increasing filter sizes (32 → 64 → 128)
- Batch normalization and dropout for regularization
- Dense layers for classification

### EfficientNet Model

A transfer learning approach using EfficientNetB0:
- Pretrained convolutional base (frozen by default)
- Custom classification head
- Option to unfreeze base model for fine-tuning

### Transformer Model

A Vision Transformer (ViT) inspired model:
- Image patches processed with positional embeddings
- Self-attention mechanisms for contextual understanding
- MLP head for classification

## Performance

The models are evaluated using the following metrics:
- Accuracy
- Precision, Recall, and F1-score (per class and macro/weighted average)
- Confusion matrix
- Visualization of best and worst-performing classes

## Future Improvements

This project is in its early stages. Planned improvements include:

- Expanding the database of hieroglyph meanings and phonetic values
- Implementing more advanced hieroglyph detection algorithms
- Adding context-aware translation by understanding hieroglyph sequences
- Developing a system that can interpret complete hieroglyphic texts
- Incorporating historical and cultural context to improve translations
- Building a mobile application for on-site archaeological use
- Adding support for other ancient scripts (Sumerian, Mayan, etc.)

## Credits

- Dataset: Morris Franken (ACM Conference on Multimedia, 2013)
- Original hieroglyphic texts: "The Pyramid of Unas" (Alexandre Piankoff, 1955)
- Gardiner Sign List for hieroglyph categorization
- Inspiration: Graham Hancock, Randall Carlson, and others exploring ancient civilizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or dataset in your research, please cite:

```
Franken, M., & van Gemert, J. C. (2013). Automatic Egyptian Hieroglyph Recognition by Retrieving Images as Texts. 
In Proceedings of the 21st ACM International Conference on Multimedia (pp. 765-768). ACM.
```
