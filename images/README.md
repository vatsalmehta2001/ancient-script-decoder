# Images for Documentation

This directory contains the following images for the project documentation:

## Existing Images (Already Copied)

I've copied several visualization images from the project:

1. **sample_hieroglyphs.png** - Grid of sample hieroglyphs from the training dataset with their Gardiner codes
   - From the newer advanced model with ~80% accuracy

2. **augmentation_examples.png** - Examples of data augmentation applied to hieroglyph images
   - From the newer advanced model with ~80% accuracy

3. **class_distribution.png** - Bar chart showing the distribution of hieroglyph classes in the dataset
   - From the newer advanced model with ~80% accuracy

4. **example_recognition.png** - Example of successful hieroglyph recognition 
   - From the older model (not the newer 80% accuracy model)

5. **model_architecture.png** - Training history graph showing accuracy and loss
   - Note: This is from the older model, not the newer 80% accuracy model
   - This is not an actual architecture diagram, but a training history plot

## Still Needed

1. **demo_screenshot.png** - Screenshot of the application's main interface showing the newer model with 80% accuracy in action

## How to Create Better Images

1. Take a screenshot of the running app with the newer model (`streamlit run app.py`)
2. Consider replacing example_recognition.png and model_architecture.png with images from the newer model
3. Consider creating an actual architecture diagram showing the CNN + ViT + EfficientNet architecture

## Image Location Information

- The visualization images for the newer model come from:
  - `clean_repo/advanced_output/`
  - `backups/visualization/`
- The example_recognition.png and model_architecture.png are from:
  - `hieroglyph_recognition/model/`

Once these images are added, the GitHub repository will display them properly in the README.md file. 