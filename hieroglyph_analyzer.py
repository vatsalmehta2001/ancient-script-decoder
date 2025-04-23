#!/usr/bin/env python3
import os
import time
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add custom learning rate scheduler to match training environment
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay with warmup learning rate scheduler
    """
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps=0):
        super(CosineDecayWithWarmup, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        # Convert from tensor to Python int for comparison operations
        step_as_int = tf.cast(step, tf.int32)
        warmup_steps_as_int = tf.cast(self.warmup_steps, tf.int32)
        
        # Logic for warmup phase
        is_warmup = tf.cast(step_as_int < warmup_steps_as_int, tf.float32)
        warmup_percent = tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.initial_learning_rate * warmup_percent
        
        # Logic for cosine decay phase
        decay_steps_adjusted = self.decay_steps - self.warmup_steps
        decay_step_adjusted = tf.cast(step - self.warmup_steps, tf.float32)
        cosine_decay = 0.5 * (1 + tf.cos(
            tf.constant(np.pi) * decay_step_adjusted / decay_steps_adjusted))
        cosine_lr = self.initial_learning_rate * cosine_decay
        
        # Choose between warmup and cosine based on step
        lr = is_warmup * warmup_lr + (1.0 - is_warmup) * cosine_lr
        return lr
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps
        }

class HieroglyphAnalyzer:
    """
    Integrated class for hieroglyph detection and recognition.
    Combines detection of hieroglyphs in images with classification
    using a pre-trained model.
    """
    
    def __init__(self, 
                 model_path="advanced_output/app_ready_model.h5",
                 class_map_path="advanced_output/class_mapping.json",
                 detection_confidence=0.5,
                 recognition_threshold=0.3,
                 input_size=(224, 224),
                 enable_debug=False):
        """
        Initialize the hieroglyph analyzer.
        
        Args:
            model_path: Path to the trained recognition model
            class_map_path: Path to class mapping JSON file
            detection_confidence: Threshold for detection confidence
            recognition_threshold: Threshold for recognition confidence
            input_size: Input size for the model (width, height)
            enable_debug: Enable debug mode with extra logging
        """
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.detection_confidence = detection_confidence
        self.recognition_threshold = recognition_threshold
        self.input_size = input_size
        self.debug = enable_debug
        
        # Initialize model and class mappings
        self._load_model()
        self._load_class_mappings()
        
        # Parameters for sliding window detection
        self.detection_params = {
            'window_sizes': [(64, 64), (96, 96), (128, 128), (160, 160)],
            'stride_factor': 0.5,  # Stride as fraction of window size
            'iou_threshold': 0.3,  # For non-maximum suppression
            'max_detections': 30    # Maximum number of detections to return
        }
        
        if self.debug:
            print(f"Initialized HieroglyphAnalyzer with model: {model_path}")
            print(f"Detection confidence: {detection_confidence}")
            print(f"Recognition threshold: {recognition_threshold}")
    
    def _load_model(self):
        """Load the recognition model from file"""
        try:
            # Register custom objects for model loading
            custom_objects = {
                'CosineDecayWithWarmup': CosineDecayWithWarmup
            }
            
            # Load model with custom objects
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            
            if self.debug:
                print(f"Successfully loaded model from {self.model_path}")
                print(f"Model summary: {self.model.summary()}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _load_class_mappings(self):
        """Load class mappings from JSON file"""
        try:
            with open(self.class_map_path, 'r') as f:
                mapping_data = json.load(f)
            
            # Check if the file has nested structure with 'class_mapping' key
            if 'class_mapping' in mapping_data:
                self.class_mapping = mapping_data['class_mapping']
                # We can use the predefined inverse mapping if it exists
                if 'inv_class_mapping' in mapping_data:
                    self.inv_class_mapping = {int(k): v for k, v in mapping_data['inv_class_mapping'].items()}
                else:
                    # Create inverse mapping for prediction
                    self.inv_class_mapping = {int(v): k for k, v in self.class_mapping.items()}
            else:
                # Assume the file directly contains the class mapping
                self.class_mapping = mapping_data
                # Create inverse mapping for prediction
                self.inv_class_mapping = {int(v): k for k, v in self.class_mapping.items()}
            
            if self.debug:
                print(f"Loaded {len(self.class_mapping)} classes from {self.class_map_path}")
                print(f"First 5 classes: {list(self.class_mapping.items())[:5]}")
                print(f"First 5 inverse mappings: {list(self.inv_class_mapping.items())[:5]}")
        except Exception as e:
            print(f"Error loading class mappings: {str(e)}")
            raise
    
    def get_gardiner_info(self, gardiner_code):
        """
        Get description information for a Gardiner code.
        
        Args:
            gardiner_code: The Gardiner classification code
            
        Returns:
            Dictionary with name and description
        """
        # Basic descriptions for common Gardiner categories
        if gardiner_code.startswith("Unknown-"):
            return {
                "name": f"Unknown Hieroglyph ({gardiner_code})",
                "description": "This hieroglyph couldn't be matched to a known classification."
            }
        
        category_mapping = {
            'A': "Man and his occupations",
            'B': "Woman and her occupations",
            'C': "Anthropomorphic deities",
            'D': "Parts of the human body",
            'E': "Mammals",
            'F': "Parts of mammals",
            'G': "Birds",
            'H': "Parts of birds",
            'I': "Amphibious animals, reptiles",
            'K': "Fish and parts of fish",
            'L': "Invertebrates and lesser animals",
            'M': "Trees and plants",
            'N': "Sky, earth, water",
            'O': "Buildings and parts of buildings",
            'P': "Ships and parts of ships",
            'Q': "Domestic and funerary furniture",
            'R': "Temple furniture and sacred emblems",
            'S': "Crowns, dress, staves",
            'T': "Warfare, hunting, butchery",
            'U': "Agriculture, crafts, professions",
            'V': "Rope, fiber, baskets, bags",
            'W': "Vessels of stone and earthenware",
            'X': "Loaves and cakes",
            'Y': "Writings, games, music",
            'Z': "Strokes, signs derived from hieratic, geometrical figures"
        }
        
        # Get the category letter
        category = gardiner_code[0] if gardiner_code else "?"
        
        # Get the category description
        category_desc = category_mapping.get(category, "Miscellaneous symbols")
        
        return {
            "name": f"Gardiner {gardiner_code}",
            "description": f"Category {category}: {category_desc}"
        }
    
    def preprocess_image(self, image):
        """
        Preprocess image for the model.
        
        Args:
            image: NumPy array of image or path to image file
            
        Returns:
            Preprocessed image as NumPy array
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in RGB format and has 3 channels
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Preprocess image for model input
        img = cv2.resize(image, self.input_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def predict_hieroglyph(self, image):
        """
        Predict the hieroglyph class from an image.
        
        Args:
            image: NumPy array of image
            
        Returns:
            List of prediction dictionaries sorted by confidence
        """
        # Preprocess the image
        processed_img = self.preprocess_image(image)
        
        # Make prediction
        prediction = self.model.predict(processed_img)
        
        # Get top-k predictions
        k = min(5, prediction.shape[1])  # Get top 5 or fewer if fewer classes
        indices = np.argsort(prediction[0])[::-1][:k]
        confidences = prediction[0][indices]
        
        results = []
        for i, idx in enumerate(indices):
            confidence = float(confidences[i])
            if confidence >= self.recognition_threshold:
                try:
                    # Convert numpy int to Python int to avoid issues
                    class_idx = int(idx)
                    if class_idx in self.inv_class_mapping:
                        class_name = self.inv_class_mapping[class_idx]
                    else:
                        class_name = f"Unknown-{class_idx}"
                    
                    # Get additional info about the hieroglyph
                    info = self.get_gardiner_info(class_name)
                    
                    results.append({
                        'class_index': class_idx,
                        'class_name': class_name,
                        'confidence': confidence,
                        'info': info
                    })
                except Exception as e:
                    if self.debug:
                        print(f"Error in class mapping for index {idx}: {str(e)}")
        
        return results
    
    def detect_hieroglyphs(self, image):
        """
        Detect potential hieroglyphs in an image using the model itself.
        
        Args:
            image: NumPy array of image or path to image file
            
        Returns:
            List of dictionaries with detected regions
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in RGB format and has 3 channels
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Make a copy of the image for processing
        original = image.copy()
        h, w = original.shape[:2]
        
        # Store detected regions and window information
        regions = []
        region_id = 0
        all_windows = []
        all_positions = []
        
        if self.debug:
            print(f"Scanning image of size {w}x{h} with sliding windows...")
        
        # Generate windows of different sizes
        for window_size in self.detection_params['window_sizes']:
            win_w, win_h = window_size
            stride_w, stride_h = int(win_w * self.detection_params['stride_factor']), int(win_h * self.detection_params['stride_factor'])
            
            # Skip if window is larger than image
            if win_w > w or win_h > h:
                continue
            
            if self.debug:
                print(f"  Processing window size {win_w}x{win_h} with stride {stride_w}x{stride_h}")
                
            # Generate window positions
            positions = []
            windows = []
            
            for y in range(0, h - win_h + 1, stride_h):
                for x in range(0, w - win_w + 1, stride_w):
                    # Extract window
                    window = original[y:y+win_h, x:x+win_w]
                    
                    # Skip windows that are too small
                    if window.size == 0:
                        continue
                    
                    # Save window and its position
                    windows.append(window)
                    positions.append((x, y, win_w, win_h))
            
            if windows:
                all_windows.extend(windows)
                all_positions.extend(positions)
        
        # Process windows in batches for efficiency
        if all_windows:
            batch_size = 16  # Process this many windows at a time
            num_batches = (len(all_windows) + batch_size - 1) // batch_size
            
            if self.debug:
                print(f"Processing {len(all_windows)} windows in {num_batches} batches...")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_windows))
                
                # Get current batch
                batch_windows = all_windows[start_idx:end_idx]
                batch_positions = all_positions[start_idx:end_idx]
                
                # Predict each window individually
                for window, position in zip(batch_windows, batch_positions):
                    predictions = self.predict_hieroglyph(window)
                    
                    # If we have a confident prediction, this may be a hieroglyph
                    if predictions and predictions[0]['confidence'] > self.detection_confidence:
                        x, y, win_w, win_h = position
                        
                        # Add as detected region
                        region = {
                            'id': region_id,
                            'bounding_box': position,
                            'confidence': predictions[0]['confidence'],
                            'roi': window,
                            'roi_padded': window,  # Same as roi for sliding window
                            'predictions': predictions
                        }
                        regions.append(region)
                        region_id += 1
        
        if self.debug:
            print(f"Found {len(regions)} potential hieroglyphs before NMS")
        
        # Apply non-maximum suppression to remove overlapping regions
        if regions:
            # Convert regions to appropriate format for NMS
            boxes = np.array([list(r['bounding_box']) for r in regions])
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 0] + boxes[:, 2]
            y2 = boxes[:, 1] + boxes[:, 3]
            scores = np.array([r['confidence'] for r in regions])
            
            # Convert to tensorflow format: [y1, x1, y2, x2]
            tf_boxes = np.column_stack([y1, x1, y2, x2])
            
            # Apply NMS
            indices = tf.image.non_max_suppression(
                tf_boxes, 
                scores, 
                max_output_size=self.detection_params['max_detections'],
                iou_threshold=self.detection_params['iou_threshold']
            ).numpy()
            
            # Filter regions using indices
            filtered_regions = [regions[i] for i in indices]
            
            if self.debug:
                print(f"After NMS: {len(filtered_regions)} hieroglyphs detected")
            
            return filtered_regions
        
        return regions
    
    def analyze_image(self, image, visualize=False):
        """
        Full analysis pipeline: detection + recognition.
        
        Args:
            image: Path to image file or NumPy array
            visualize: Whether to include visualization in results
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            file_info = {'filename': os.path.basename(image), 'path': image}
        else:
            img = image.copy()
            file_info = {'filename': 'array_input', 'path': None}
        
        # Ensure image is in RGB format
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Detection
        detection_start = time.time()
        regions = self.detect_hieroglyphs(img)
        detection_time = time.time() - detection_start
        
        # Results are already available from detection since we're using the model
        # for both detection and recognition
        results = []
        for region in regions:
            # Add the prediction results to the region
            result = {
                'id': region['id'],
                'bounding_box': region['bounding_box'],
                'predictions': region['predictions']
            }
            results.append(result)
        
        recognition_time = 0  # Recognition already done in detection phase
        total_time = time.time() - start_time
        
        # Create the result object
        analysis_results = {
            'file_info': file_info,
            'total_time_sec': total_time,
            'detection': {
                'time_sec': detection_time,
                'count': len(regions)
            },
            'recognition': {
                'time_sec': recognition_time,
                'count': len(results)
            },
            'results': results
        }
        
        # Add visualization if requested
        if visualize:
            analysis_results['visualization'] = self.visualize_results(img, results)
        
        return analysis_results
    
    def visualize_results(self, image, results):
        """
        Create a visualization of the analysis results.
        
        Args:
            image: The original image as NumPy array
            results: List of result dictionaries
            
        Returns:
            Visualization image as NumPy array
        """
        # Make a copy of the image for visualization
        vis_image = image.copy()
        
        # Define colors
        box_color = (0, 255, 0)  # Green for bounding boxes
        text_color = (255, 255, 255)  # White for text
        
        # Add bounding boxes and labels
        for result in results:
            # Get bounding box
            x, y, w, h = result['bounding_box']
            
            # Get top prediction
            if result['predictions']:
                top_pred = result['predictions'][0]
                confidence = top_pred['confidence']
                
                # Set colors based on confidence
                if confidence > 0.7:
                    box_color = (0, 255, 0)  # Strong green for high confidence
                elif confidence > 0.5:
                    box_color = (0, 200, 200)  # Yellow-green for medium confidence
                else:
                    box_color = (0, 165, 255)  # Orange for lower confidence
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), box_color, 2)
                
                # Create the label text
                class_name = top_pred['class_name']
                label = f"{class_name} ({confidence:.2f})"
                
                # Calculate text position and size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                
                # Draw label background - larger for better visibility
                cv2.rectangle(
                    vis_image, 
                    (x, y - text_size[1] - 10), 
                    (x + text_size[0] + 10, y), 
                    (0, 0, 0), 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image, 
                    label, 
                    (x + 5, y - 5), 
                    font, 
                    font_scale, 
                    text_color, 
                    font_thickness
                )
        
        return vis_image
    
    def batch_process(self, input_dir, output_dir, 
                      save_visualizations=True, save_json=True,
                      extensions=None):
        """
        Process a directory of images.
        
        Args:
            input_dir: Directory containing images to process
            output_dir: Directory to save results
            save_visualizations: Whether to save visualization images
            save_json: Whether to save JSON results
            extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
            
        Returns:
            List of result dictionaries
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png']
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        image_files = []
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(file_path)
        
        if self.debug:
            print(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files):
            try:
                # Get base filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                if self.debug:
                    print(f"Processing {i+1}/{len(image_files)}: {base_name}")
                
                # Analyze the image
                result = self.analyze_image(image_path, visualize=save_visualizations)
                
                # Save visualization if requested
                if save_visualizations and 'visualization' in result:
                    vis_path = os.path.join(output_dir, f"{base_name}_analyzed.jpg")
                    cv2.imwrite(vis_path, cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR))
                
                # Save JSON results if requested
                if save_json:
                    # Remove large objects before saving to JSON
                    result_copy = result.copy()
                    if 'visualization' in result_copy:
                        del result_copy['visualization']
                    
                    # Remove numpy arrays (ROIs) from regions
                    for item in result_copy.get('results', []):
                        if 'roi' in item:
                            del item['roi']
                        if 'roi_padded' in item:
                            del item['roi_padded']
                    
                    # Save JSON
                    json_path = os.path.join(output_dir, f"{base_name}_results.json")
                    with open(json_path, 'w') as f:
                        json.dump(result_copy, f, indent=2)
                
                # Add result to list
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'file_info': {'filename': os.path.basename(image_path), 'path': image_path},
                    'error': str(e)
                })
        
        return results 