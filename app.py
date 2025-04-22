import streamlit as st
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from PIL import Image
import cv2
import pandas as pd
from preprocessing import HieroglyphDataProcessor
import io
import base64
from pathlib import Path

class HieroglyphDecoder:
    def __init__(self, model_path, class_mapping_path):
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_mapping_data = json.load(f)
        
        self.inv_class_mapping = {int(k): v for k, v in class_mapping_data['inv_class_mapping'].items()}
        self.class_mapping = class_mapping_data['class_mapping']
        
        # Image size
        self.img_size = (75, 50)  # width, height
        
        # Create instances directory for saving uploaded images
        self.instances_dir = Path('uploaded_instances')
        os.makedirs(self.instances_dir, exist_ok=True)
    
    def preprocess_image(self, image, normalize=True):
        """Preprocess image for prediction"""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
            
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # Handle RGBA
            img = img[:, :, :3]
            
        # Resize to target size
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        
        # Normalize if needed
        if normalize:
            img = img.astype(np.float32) / 255.0
            
        return img
    
    def predict_hieroglyph(self, image):
        """Predict hieroglyph class for a given image"""
        # Preprocess image
        img = self.preprocess_image(image)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Get prediction
        pred = self.model.predict(img)[0]
        
        # Get top 5 predictions
        top_indices = pred.argsort()[-5:][::-1]
        top_preds = [{
            'gardiner_code': self.inv_class_mapping[idx],
            'confidence': float(pred[idx])
        } for idx in top_indices]
        
        return top_preds
    
    def crop_hieroglyph(self, image, box):
        """Crop hieroglyph from image using bounding box"""
        x, y, w, h = box
        if isinstance(image, Image.Image):
            cropped = image.crop((x, y, x+w, y+h))
        else:
            cropped = image[y:y+h, x:x+w]
        return cropped
    
    def detect_hieroglyphs(self, image):
        """
        Simple contour-based detection of potential hieroglyphs
        This is a very basic approach and could be improved with proper object detection
        """
        # Convert to numpy array if PIL image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
            
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 100  # Minimum area to consider
        max_area = image_np.shape[0] * image_np.shape[1] * 0.5  # Max 50% of image
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio
                aspect_ratio = w / h
                if 0.2 <= aspect_ratio <= 5:  # Only keep reasonably shaped contours
                    valid_contours.append((x, y, w, h))
        
        return valid_contours
    
    def process_image(self, image):
        """Process full image, detect hieroglyphs, and recognize them"""
        # Detect potential hieroglyphs
        hieroglyph_boxes = self.detect_hieroglyphs(image)
        
        results = []
        for i, box in enumerate(hieroglyph_boxes):
            # Crop hieroglyph
            cropped = self.crop_hieroglyph(image, box)
            
            # Save cropped image
            img_path = self.instances_dir / f"hieroglyph_{i}.png"
            if isinstance(cropped, Image.Image):
                cropped.save(img_path)
            else:
                cv2.imwrite(str(img_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
            # Predict
            predictions = self.predict_hieroglyph(cropped)
            
            results.append({
                'box': box,
                'predictions': predictions,
                'image_path': str(img_path)
            })
            
        return results
    
    def draw_detection_results(self, image, results):
        """Draw detection boxes and predictions on the image"""
        # Convert to numpy array for OpenCV operations
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
            
        # Create a copy for drawing
        annotated_img = image_np.copy()
        
        # Colors for different confidence levels
        colors = {
            'high': (0, 255, 0),  # Green for high confidence
            'medium': (0, 255, 255),  # Yellow for medium confidence
            'low': (0, 0, 255)  # Red for low confidence
        }
        
        for result in results:
            box = result['box']
            top_prediction = result['predictions'][0]
            confidence = top_prediction['confidence']
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = colors['high']
            elif confidence > 0.5:
                color = colors['medium']
            else:
                color = colors['low']
                
            # Draw rectangle
            x, y, w, h = box
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{top_prediction['gardiner_code']} ({confidence:.2f})"
            cv2.putText(annotated_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return annotated_img
        
    @staticmethod
    def get_gardiner_info(gardiner_code):
        """Get information about a Gardiner code from a predefined dictionary"""
        # This is a placeholder - in a real system, this would be a database lookup
        gardiner_info = {
            'A1': 'Seated man', 
            'D21': 'Mouth', 
            'G1': 'Egyptian vulture', 
            'M17': 'Reed', 
            'N35': 'Water', 
            'O1': 'House', 
            'Q3': 'Stool', 
            'S29': 'Folded cloth', 
            'X1': 'Bread loaf'
            # Add more entries as needed
        }
        
        return gardiner_info.get(gardiner_code, f"Information not available for {gardiner_code}")
   
# Streamlit UI
def main():
    st.set_page_config(
        page_title="Ancient Egyptian Hieroglyph Decoder",
        page_icon="🏺",
        layout="wide"
    )
    
    st.title("Ancient Egyptian Hieroglyph Decoder")
    st.markdown("""
    This application helps identify and translate ancient Egyptian hieroglyphs using deep learning.
    Upload an image containing hieroglyphs to get started!
    """)
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Model path selection
    model_path = st.sidebar.selectbox(
        "Select model",
        ["hieroglyph_recognition/model/final_model.h5", "hieroglyph_recognition/deployment/saved_model"],
        format_func=lambda x: "Default CNN Model" if "final_model" in x else "Optimized Deployment Model"
    )
    
    # Class mapping path
    class_mapping_path = "hieroglyph_recognition/class_mapping.json"
    
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(class_mapping_path):
        st.error("""
        Model files not found! Please make sure you've trained the model and have the following files:
        - hieroglyph_recognition/model/final_model.h5
        - hieroglyph_recognition/class_mapping.json
        
        Run the training script first:
        ```
        python train.py
        ```
        """)
        return
    
    # Initialize decoder
    decoder = HieroglyphDecoder(model_path, class_mapping_path)
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select mode",
        ["Single Hieroglyph", "Full Image Analysis"]
    )
    
    # File uploader
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Resize image for display if too large
        max_width = 800
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height))
        
        # Display original image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process based on mode
        if mode == "Single Hieroglyph":
            # Process single hieroglyph
            if st.button("Analyze Hieroglyph"):
                with st.spinner("Analyzing..."):
                    predictions = decoder.predict_hieroglyph(image)
                    
                    # Display results
                    st.subheader("Recognition Results")
                    
                    # Display top predictions in a table
                    results_df = pd.DataFrame(predictions)
                    results_df.columns = ['Gardiner Code', 'Confidence']
                    results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x:.2%}")
                    st.table(results_df)
                    
                    # Show information about the top prediction
                    top_code = predictions[0]['gardiner_code']
                    st.subheader(f"Information about {top_code}")
                    st.write(decoder.get_gardiner_info(top_code))
        
        else:  # Full Image Analysis
            if st.button("Detect and Analyze Hieroglyphs"):
                with st.spinner("Analyzing full image..."):
                    # Process the full image
                    results = decoder.process_image(image)
                    
                    # Draw detection results
                    if results:
                        annotated_img = decoder.draw_detection_results(image, results)
                        st.image(annotated_img, caption="Detection Results", use_column_width=True)
                        
                        # Display individual results
                        st.subheader(f"Detected {len(results)} potential hieroglyphs")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Hieroglyph {i+1} - {result['predictions'][0]['gardiner_code']}"):
                                # Create columns for image and predictions
                                col1, col2 = st.columns([1, 3])
                                
                                # Display cropped hieroglyph
                                cropped_img = Image.open(result['image_path'])
                                col1.image(cropped_img, caption=f"Cropped Hieroglyph {i+1}")
                                
                                # Display predictions
                                predictions_df = pd.DataFrame(result['predictions'])
                                predictions_df.columns = ['Gardiner Code', 'Confidence']
                                predictions_df['Confidence'] = predictions_df['Confidence'].apply(lambda x: f"{x:.2%}")
                                col2.table(predictions_df)
                                
                                # Add information about the top prediction
                                top_code = result['predictions'][0]['gardiner_code']
                                col2.markdown(f"**Information**: {decoder.get_gardiner_info(top_code)}")
                    else:
                        st.warning("No hieroglyphs detected in the image. Try adjusting the image or use a different one.")
    
    # Instructions and information section
    with st.expander("How to use this app"):
        st.markdown("""
        ## Instructions
        1. **Upload an image** containing Egyptian hieroglyphs
        2. Choose the mode:
           - **Single Hieroglyph**: For analyzing a single clear hieroglyph
           - **Full Image Analysis**: For detecting and recognizing multiple hieroglyphs in a scene
        3. Click the "Analyze" button to process the image
        4. Review the results, which include recognized hieroglyphs and their Gardiner codes
        
        ## About Gardiner Codes
        The Gardiner Sign List is a classification system for Egyptian hieroglyphs, organized by visual categories.
        Each code consists of a letter (category) and a number. For example:
        - A: Men and their occupations
        - D: Parts of the human body
        - G: Birds
        - M: Trees and plants
        - N: Sky, earth, water
        - O: Buildings and parts of buildings
        
        ## Limitations
        - This system works best with clear, high-contrast images
        - Complex hieroglyphic arrangements may not be detected accurately
        - The system recognizes individual hieroglyphs but doesn't provide full translation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Ancient Egyptian Hieroglyph Decoder | Developed with TensorFlow and Streamlit")

if __name__ == "__main__":
    main() 