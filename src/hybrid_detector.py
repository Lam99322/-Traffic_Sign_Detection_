import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from .yolo_detector import YOLOv11Detector
from .cnn_classifier import TrafficSignCNN, VisionTransformerClassifier

logger = logging.getLogger(__name__)

class HybridTrafficSignDetector:
    """
    Hybrid Traffic Sign Detection System:
    1. YOLOv11 for detection and localization
    2. CNN/ViT for classification of detected signs
    """
    
    def __init__(self, 
                 yolo_model_path: str = 'models/yolo_traffic_signs.pt',
                 cnn_model_path: Optional[str] = None,
                 vit_model_path: Optional[str] = None,
                 use_vit: bool = False,
                 confidence_threshold: float = 0.5):
        """
        Initialize hybrid detector
        
        Args:
            yolo_model_path: Path to YOLOv11 model
            cnn_model_path: Path to CNN model (optional)
            vit_model_path: Path to ViT model (optional)  
            use_vit: Whether to use ViT instead of CNN
            confidence_threshold: Minimum confidence for detections
        """
        
        self.confidence_threshold = confidence_threshold
        self.use_vit = use_vit
        
        # Initialize YOLO detector
        try:
            self.yolo_detector = YOLOv11Detector(
                model_path=yolo_model_path,
                confidence_threshold=confidence_threshold
            )
            logger.info("YOLOv11 detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            self.yolo_detector = None
        
        # Initialize classifier (CNN or ViT)
        try:
            if use_vit:
                self.classifier = VisionTransformerClassifier(model_path=vit_model_path)
                logger.info("Vision Transformer classifier initialized")
            else:
                self.classifier = TrafficSignCNN(model_path=cnn_model_path)
                logger.info("CNN classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            self.classifier = None
    
    def detect_and_classify(self, image: np.ndarray) -> Dict:
        """
        Complete pipeline: detect signs then classify them
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Dict with detections and classifications
        """
        results = {
            'detections': [],
            'classifications': [],
            'total_signs': 0,
            'processing_time': 0,
            'errors': []
        }
        
        if self.yolo_detector is None:
            results['errors'].append("YOLO detector not available")
            return results
        
        if self.classifier is None:
            results['errors'].append("Classifier not available")
            return results
        
        try:
            import time
            start_time = time.time()
            
            # Step 1: YOLO Detection
            detection_results = self.yolo_detector.detect_signs(image)
            
            if not detection_results or 'detections' not in detection_results:
                logger.warning("No traffic signs detected by YOLO")
                results['processing_time'] = time.time() - start_time
                return results
            
            # Extract detection data
            detections = detection_results['detections']
            cropped_images = detection_results.get('cropped_images', [])
            
            results['detections'] = detections
            results['total_signs'] = len(detections)
            
            if not cropped_images:
                logger.warning("No cropped images available for classification")
                results['processing_time'] = time.time() - start_time
                return results
            
            # Step 2: Classification
            logger.info(f"Classifying {len(cropped_images)} detected signs...")
            classifications = self.classifier.classify_signs(cropped_images)
            
            # Combine detection and classification results
            combined_results = []
            for i, (detection, classification) in enumerate(zip(detections, classifications)):
                if 'error' not in classification:
                    combined_result = {
                        'detection_id': i,
                        'bbox': detection['bbox'],
                        'detection_confidence': detection['confidence'],
                        'detection_class': detection.get('class', 'traffic_sign'),
                        
                        # Classification results
                        'class_id': classification['class_id'],
                        'class_name': classification['class_name'],
                        'classification_confidence': classification['confidence'],
                        'model_type': classification.get('model_type', 'CNN'),
                        
                        # Combined confidence score
                        'combined_confidence': (detection['confidence'] + classification['confidence']) / 2
                    }
                    combined_results.append(combined_result)
                else:
                    logger.error(f"Classification error for detection {i}: {classification['error']}")
            
            results['classifications'] = combined_results
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"Processed {len(combined_results)} signs in {results['processing_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Error in hybrid detection pipeline: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection boxes and classification labels on image
        
        Args:
            image: Original image
            results: Results from detect_and_classify()
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        if 'classifications' not in results or not results['classifications']:
            return annotated_image
        
        for result in results['classifications']:
            # Extract bbox coordinates
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox
            
            # Detection confidence color (green = high, red = low)
            det_conf = result['detection_confidence']
            cls_conf = result['classification_confidence']
            
            # Box color based on combined confidence
            combined_conf = result['combined_confidence']
            if combined_conf > 0.8:
                color = (0, 255, 0)  # Green
            elif combined_conf > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Prepare label text
            label = f"{result['class_name']}"
            confidence_text = f"D:{det_conf:.2f} C:{cls_conf:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            max_width = max(label_size[0], conf_size[0])
            total_height = label_size[1] + conf_size[1] + 10
            
            cv2.rectangle(annotated_image, 
                         (int(x1), int(y1) - total_height - 5), 
                         (int(x1) + max_width + 10, int(y1)), 
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_image, label, 
                       (int(x1) + 5, int(y1) - conf_size[1] - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(annotated_image, confidence_text, 
                       (int(x1) + 5, int(y1) - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_image
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        """
        info = {
            'yolo_available': self.yolo_detector is not None,
            'classifier_available': self.classifier is not None,
            'classifier_type': 'ViT' if self.use_vit else 'CNN',
            'confidence_threshold': self.confidence_threshold
        }
        
        if self.yolo_detector:
            info['yolo_device'] = self.yolo_detector.device
            info['yolo_classes'] = self.yolo_detector.num_classes
        
        if self.classifier:
            info['classifier_classes'] = self.classifier.num_classes
            info['class_names'] = self.classifier.class_names
        
        return info

# Convenience function for quick usage
def create_hybrid_detector(yolo_path: str = None, 
                         cnn_path: str = None, 
                         use_vit: bool = False,
                         confidence: float = 0.5) -> HybridTrafficSignDetector:
    """
    Create hybrid detector with default paths
    """
    return HybridTrafficSignDetector(
        yolo_model_path=yolo_path or 'models/yolo_traffic_signs.pt',
        cnn_model_path=cnn_path,
        use_vit=use_vit,
        confidence_threshold=confidence
    )