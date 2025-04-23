import streamlit as st
import os
import time
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import tensorflow as tf
from hieroglyph_analyzer import HieroglyphAnalyzer, CosineDecayWithWarmup

# Configure page
st.set_page_config(
    page_title="Ancient Egyptian Hieroglyph Analyzer",
    page_icon="🏺",
    layout="wide"
)

# CSS for improved appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3d5a80;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #293241;
        margin-top: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3d5a80;
    }
    .confidence-high {
        color: #2a9d8f;
        font-weight: bold;
    }
    .confidence-medium {
        color: #e9c46a;
        font-weight: bold;
    }
    .confidence-low {
        color: #e76f51;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize analyzer
@st.cache_resource
def load_analyzer(detection_confidence=0.5, recognition_threshold=0.3, debug=False):
    analyzer = HieroglyphAnalyzer(
        model_path="advanced_output/app_ready_model.h5",
        class_map_path="advanced_output/class_mapping.json",
        detection_confidence=detection_confidence,
        recognition_threshold=recognition_threshold,
        enable_debug=debug
    )
    return analyzer

# Initialize session state
if 'full_analysis_results' not in st.session_state:
    st.session_state.full_analysis_results = None
if 'single_analysis_results' not in st.session_state:
    st.session_state.single_analysis_results = None
if 'image' not in st.session_state:
    st.session_state.image = None

# Setup directories
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("analyzer_output", exist_ok=True)

# App title
st.markdown("<h1 class='main-header'>Ancient Egyptian Hieroglyph Analyzer</h1>", unsafe_allow_html=True)

# Mode selection
mode = st.radio("Select Analysis Mode:", ["Single Hieroglyph Analysis", "Full Image Analysis"])

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    detection_confidence = st.slider(
        "Detection Confidence", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5,
        help="Minimum confidence for hieroglyph detection"
    )
    
    recognition_threshold = st.slider(
        "Recognition Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.3,
        help="Minimum confidence for recognition"
    )
    
    debug_mode = st.checkbox("Debug Mode", value=False)
    
    st.markdown("---")
    st.markdown("""
    **Model Info**  
    - Architecture: Advanced CNN + EfficientNet + ViT
    - Accuracy: 82.66% (Top-1), 95% (Top-3)
    """)

# Upload image
uploaded_file = st.file_uploader("Upload an image containing hieroglyphs", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save the image
    img_path = os.path.join("uploaded_images", uploaded_file.name)
    with open(img_path, "wb") as f:
        image.save(f, format="JPEG")
    
    st.session_state.image = image
    
    # Add example images option
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Use a test image instead"):
            test_images = [f for f in os.listdir("test_images") if f.endswith(('.jpg', '.jpeg', '.png'))]
            if test_images:
                img_path = os.path.join("test_images", test_images[0])
                image = Image.open(img_path).convert('RGB')  # Convert to RGB
                st.session_state.image = image
                st.image(image, caption="Test Image", use_container_width=True)
    
    analyze_button = st.button("Analyze Image")
    
    if analyze_button and st.session_state.image is not None:
        # Load analyzer
        analyzer = load_analyzer(
            detection_confidence=detection_confidence,
            recognition_threshold=recognition_threshold,
            debug=debug_mode
        )
        
        # Convert image for analysis
        img_array = np.array(st.session_state.image.convert('RGB'))  # Ensure RGB format
        
        with st.spinner("Analyzing image... This may take a moment."):
            if mode == "Single Hieroglyph Analysis":
                # For single hieroglyph, use predict_hieroglyph directly
                predictions = analyzer.predict_hieroglyph(img_array)
                st.session_state.single_analysis_results = predictions
                
                # Display results
                st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
                
                if predictions:
                    for i, pred in enumerate(predictions):
                        confidence = pred['confidence']
                        confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.5 else "confidence-low"
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>{i+1}. {pred['class_name']}</h3>
                            <p>Confidence: <span class="{confidence_class}">{confidence:.2f}</span></p>
                            <p><strong>Description:</strong> {pred['info']['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No hieroglyphs recognized in this image. Try adjusting the confidence threshold.")
                    
            else:  # Full Image Analysis
                # Run full analysis including detection
                analysis_results = analyzer.analyze_image(img_array, visualize=True)
                st.session_state.full_analysis_results = analysis_results
                
                # Display summary
                st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Hieroglyphs", analysis_results['detection']['count'])
                with col2:
                    st.metric("Processing Time", f"{analysis_results['total_time_sec']:.2f} sec")
                
                # Display visualization
                if 'visualization' in analysis_results:
                    st.image(analysis_results['visualization'], caption="Analysis Visualization", use_container_width=True)
                
                # Show detailed results
                if analysis_results['results']:
                    # Convert to DataFrame for display
                    results_data = []
                    for i, result in enumerate(analysis_results['results']):
                        if result['predictions']:
                            top_pred = result['predictions'][0]
                            results_data.append({
                                "ID": i+1,
                                "Hieroglyph": top_pred['class_name'],
                                "Confidence": f"{top_pred['confidence']:.2f}",
                                "Description": top_pred['info']['description']
                            })
                    
                    if results_data:
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Display individual hieroglyphs
                        st.markdown("<h3 class='sub-header'>Individual Hieroglyphs</h3>", unsafe_allow_html=True)
                        
                        # Display in a grid
                        cols = st.columns(3)
                        for i, result in enumerate(analysis_results['results']):
                            if result['predictions']:
                                col_idx = i % 3
                                with cols[col_idx]:
                                    # Extract the hieroglyph from the original image
                                    x, y, w, h = result['bounding_box']
                                    img_array = np.array(st.session_state.image)
                                    
                                    # Ensure coordinates are within image bounds
                                    h_img, w_img = img_array.shape[:2]
                                    x = max(0, min(x, w_img-1))
                                    y = max(0, min(y, h_img-1))
                                    w = min(w, w_img-x)
                                    h = min(h, h_img-y)
                                    
                                    # Extract the region
                                    try:
                                        hieroglyph_img = img_array[y:y+h, x:x+w]
                                        
                                        # Display details in a card-like format
                                        top_pred = result['predictions'][0]
                                        confidence = top_pred['confidence']
                                        confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.5 else "confidence-low"
                                        
                                        st.markdown(f"""
                                        <div class="result-box">
                                            <h4>#{i+1}: {top_pred['class_name']}</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        if hieroglyph_img.size > 0:
                                            # Convert to RGB if needed
                                            if hieroglyph_img.ndim > 2 and hieroglyph_img.shape[2] == 4:
                                                hieroglyph_img = cv2.cvtColor(hieroglyph_img, cv2.COLOR_RGBA2RGB)
                                            elif hieroglyph_img.ndim == 2:
                                                hieroglyph_img = cv2.cvtColor(hieroglyph_img, cv2.COLOR_GRAY2RGB)
                                                
                                            st.image(hieroglyph_img, width=150)
                                        
                                        st.markdown(f"""
                                        <p>Confidence: <span class="{confidence_class}">{confidence:.2f}</span></p>
                                        <p><strong>Description:</strong> {top_pred['info']['description']}</p>
                                        """, unsafe_allow_html=True)
                                        
                                    except Exception as e:
                                        if debug_mode:
                                            st.error(f"Error displaying hieroglyph #{i+1}: {str(e)}")
                else:
                    st.warning("No hieroglyphs were detected in the image. Try adjusting the detection parameters or use a different image.")
else:
    st.info("Please upload an image to begin analysis.")
    
    # Show example section
    st.markdown("<h3 class='sub-header'>Or try with an example image:</h3>", unsafe_allow_html=True)
    
    test_images = [f for f in os.listdir("test_images") if f.endswith(('.jpg', '.jpeg', '.png'))]
    if test_images:
        cols = st.columns(3)
        for i, example in enumerate(test_images[:3]):
            with cols[i % 3]:
                img_path = os.path.join("test_images", example)
                st.image(img_path, caption=example, use_container_width=True)
                
                if st.button(f"Use this image", key=f"example_{i}"):
                    image = Image.open(img_path).convert('RGB')  # Convert to RGB
                    st.session_state.image = image
                    st.rerun()  # Use st.rerun() instead of experimental_rerun

# Information section at the bottom
with st.expander("About the Gardiner Sign List"):
    st.write("""
    The hieroglyphs detected by this application are classified according to the Gardiner Sign List, 
    a standard catalog of Egyptian hieroglyphs organized by visual categories. 
    Sir Alan Gardiner, a British Egyptologist, created this system in the early 20th century.
    
    The list organizes hieroglyphs into 26 categories from A to Z, including categories such as:
    - **A**: Man and his occupations
    - **D**: Parts of the human body
    - **G**: Birds
    - **M**: Trees and plants
    - **N**: Sky, earth, water
    - And many more...
    """) 