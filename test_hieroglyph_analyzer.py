#!/usr/bin/env python3
import os
import argparse
import time
from hieroglyph_analyzer import HieroglyphAnalyzer

def main():
    """Test the HieroglyphAnalyzer class"""
    parser = argparse.ArgumentParser(description="Test HieroglyphAnalyzer")
    parser.add_argument("--image", type=str, help="Path to a single test image")
    parser.add_argument("--dir", type=str, default="./test_images", help="Directory of test images")
    parser.add_argument("--model", type=str, default="./advanced_output/app_ready_model.h5", 
                        help="Path to model file")
    parser.add_argument("--output", type=str, default="./analyzer_output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the analyzer
    print(f"Initializing HieroglyphAnalyzer with model: {args.model}")
    try:
        analyzer = HieroglyphAnalyzer(
            model_path=args.model,
            class_map_path="./advanced_output/class_mapping.json",
            enable_debug=args.debug
        )
        print("Successfully initialized analyzer")
    except Exception as e:
        print(f"Failed to initialize analyzer: {str(e)}")
        return
    
    # Process a single image if provided
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
        
        try:
            print(f"Analyzing single image: {args.image}")
            start_time = time.time()
            results = analyzer.analyze_image(args.image, visualize=True)
            elapsed = time.time() - start_time
            
            print(f"Analysis completed in {elapsed:.2f} seconds")
            print(f"Detected {results['detection']['count']} potential hieroglyphs")
            print(f"Successfully recognized {results['recognition']['count']} hieroglyphs")
            
            # Print recognized hieroglyphs
            if results['results']:
                print("\nRecognized Hieroglyphs:")
                for i, result in enumerate(results['results']):
                    if result['predictions']:
                        top_pred = result['predictions'][0]
                        print(f"  {i+1}. {top_pred['class_name']} - {top_pred['confidence']:.2f}")
                        print(f"     {top_pred['info']['description']}")
            
            # Save results
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            
            # Save visualization
            if 'visualization' in results:
                import cv2
                vis_path = os.path.join(args.output, f"{base_name}_analyzed.jpg")
                # Convert RGB to BGR for OpenCV
                vis_bgr = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_path, vis_bgr)
                print(f"Saved visualization to {vis_path}")
                
                # Display the image if possible
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 8))
                    plt.imshow(results['visualization'])
                    plt.title(f"Analyzed: {base_name} - Found {len(results['results'])} hieroglyphs")
                    
                    # Add text about detection parameters
                    params_text = (
                        f"Detection Parameters:\n"
                        f"Window Sizes: {analyzer.detection_params['window_sizes']}\n"
                        f"Stride Factor: {analyzer.detection_params['stride_factor']}\n"
                        f"IoU Threshold: {analyzer.detection_params['iou_threshold']}\n"
                        f"Max Detections: {analyzer.detection_params['max_detections']}"
                    )
                    plt.figtext(0.02, 0.02, params_text, 
                               bbox=dict(facecolor='white', alpha=0.8),
                               fontsize=9)
                    
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Could not display image: {str(e)}")
        
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Process a directory of images
    elif args.dir and os.path.isdir(args.dir):
        try:
            print(f"Processing images in directory: {args.dir}")
            start_time = time.time()
            
            results = analyzer.batch_process(
                args.dir,
                args.output,
                save_visualizations=True,
                save_json=True
            )
            
            elapsed = time.time() - start_time
            print(f"Processed {len(results)} images in {elapsed:.2f} seconds")
            print(f"Results saved to {args.output}")
            
        except Exception as e:
            print(f"Error processing directory: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        print("Please provide either an image file with --image or a directory with --dir")
        parser.print_help()

if __name__ == "__main__":
    main() 