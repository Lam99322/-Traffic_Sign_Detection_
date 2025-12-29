"""
Demo and testing script for hybrid YOLOv11 + CNN/ViT traffic sign detection
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from hybrid_detector import HybridTrafficSignDetector

def test_image(image_path: str, use_vit: bool = False, show_results: bool = True):
    """
    Test hybrid detector on a single image
    
    Args:
        image_path: Path to test image
        use_vit: Whether to use ViT instead of CNN
        show_results: Whether to display results
    """
    print(f"Testing hybrid detector on: {image_path}")
    print(f"Using classifier: {'ViT' if use_vit else 'CNN'}")
    
    # Initialize detector
    detector = HybridTrafficSignDetector(
        confidence_threshold=0.5,
        use_vit=use_vit
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Run detection and classification
    results = detector.detect_and_classify(image)
    
    # Print results
    print(f"\n=== DETECTION RESULTS ===")
    print(f"Total signs detected: {results['total_signs']}")
    print(f"Processing time: {results['processing_time']:.3f}s")
    
    if results['errors']:
        print(f"Errors: {results['errors']}")
    
    if results['classifications']:
        print(f"\n=== CLASSIFICATION RESULTS ===")
        for i, result in enumerate(results['classifications']):
            print(f"\nSign {i+1}:")
            print(f"  Location: {result['bbox']}")
            print(f"  Class: {result['class_name']} ({result['class_id']})")
            print(f"  Detection confidence: {result['detection_confidence']:.3f}")
            print(f"  Classification confidence: {result['classification_confidence']:.3f}")
            print(f"  Combined confidence: {result['combined_confidence']:.3f}")
            print(f"  Model type: {result.get('model_type', 'Unknown')}")
    
    # Visualize results
    if show_results:
        annotated_image = detector.visualize_results(image, results)
        
        # Display image
        cv2.imshow('Traffic Sign Detection Results', annotated_image)
        print(f"\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_path = f"test_result_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"Result saved to: {output_path}")
    
    # Print model info
    print(f"\n=== MODEL INFORMATION ===")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")

def test_webcam(use_vit: bool = False):
    """
    Test hybrid detector with webcam
    
    Args:
        use_vit: Whether to use ViT instead of CNN
    """
    print(f"Starting webcam test with {'ViT' if use_vit else 'CNN'} classifier")
    
    # Initialize detector
    detector = HybridTrafficSignDetector(
        confidence_threshold=0.5,
        use_vit=use_vit
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam")
            break
        
        frame_count += 1
        
        # Process every 10th frame to reduce load
        if frame_count % 10 == 0:
            # Run detection
            results = detector.detect_and_classify(frame)
            
            # Visualize
            annotated_frame = detector.visualize_results(frame, results)
            
            # Add info text
            info_text = f"Signs: {results['total_signs']}, Time: {results['processing_time']:.3f}s"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Webcam Traffic Sign Detection', annotated_frame)
        else:
            cv2.imshow('Webcam Traffic Sign Detection', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"webcam_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed")

def benchmark_models():
    """
    Compare CNN vs ViT performance on sample images
    """
    print("Benchmarking CNN vs ViT classifiers...")
    
    # Test images (you can add your own)
    test_images = [
        "data/sample/test_image_1.jpg",
        "data/sample/test_image_2.jpg"
    ]
    
    results_comparison = []
    
    for image_path in test_images:
        if not Path(image_path).exists():
            print(f"Skipping {image_path} - file not found")
            continue
        
        print(f"\nTesting: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {image_path}")
            continue
        
        # Test with CNN
        print("  Testing with CNN...")
        cnn_detector = HybridTrafficSignDetector(use_vit=False)
        cnn_results = cnn_detector.detect_and_classify(image)
        
        # Test with ViT  
        print("  Testing with ViT...")
        vit_detector = HybridTrafficSignDetector(use_vit=True)
        vit_results = vit_detector.detect_and_classify(image)
        
        # Compare results
        comparison = {
            'image': image_path,
            'cnn_time': cnn_results['processing_time'],
            'vit_time': vit_results['processing_time'],
            'cnn_detections': len(cnn_results['classifications']),
            'vit_detections': len(vit_results['classifications'])
        }
        
        results_comparison.append(comparison)
        
        print(f"  CNN: {comparison['cnn_detections']} signs, {comparison['cnn_time']:.3f}s")
        print(f"  ViT: {comparison['vit_detections']} signs, {comparison['vit_time']:.3f}s")
    
    # Summary
    print(f"\n=== BENCHMARK SUMMARY ===")
    for result in results_comparison:
        print(f"Image: {Path(result['image']).name}")
        print(f"  CNN: {result['cnn_detections']} signs, {result['cnn_time']:.3f}s")
        print(f"  ViT: {result['vit_detections']} signs, {result['vit_time']:.3f}s")
        if result['cnn_time'] > 0 and result['vit_time'] > 0:
            speedup = result['vit_time'] / result['cnn_time']
            print(f"  CNN is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than ViT")

def main():
    """Main demo script"""
    parser = argparse.ArgumentParser(description="Demo hybrid traffic sign detector")
    parser.add_argument("--mode", choices=['image', 'webcam', 'benchmark'], 
                       default='image', help="Demo mode")
    parser.add_argument("--image", help="Path to test image (for image mode)")
    parser.add_argument("--vit", action='store_true', help="Use ViT instead of CNN")
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        if not args.image:
            # Use default test image
            test_images = [
                "data/sample/test_image.jpg",
                "static/sample_image.jpg",
                "test_image.jpg"
            ]
            
            image_path = None
            for img in test_images:
                if Path(img).exists():
                    image_path = img
                    break
            
            if image_path is None:
                print("No test image found. Please specify --image path")
                print("Looking for: " + ", ".join(test_images))
                return
        else:
            image_path = args.image
        
        test_image(image_path, use_vit=args.vit)
        
    elif args.mode == 'webcam':
        test_webcam(use_vit=args.vit)
        
    elif args.mode == 'benchmark':
        benchmark_models()

if __name__ == "__main__":
    main()