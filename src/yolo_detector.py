


import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class YOLOv11Detector:
    """
    YOLOv11 Traffic Sign Detector
    Specialized for detecting traffic sign locations in images
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """
        Initialize YOLOv11 detector
        
        Args:
            model_path: Path to trained YOLOv11 model (.pt file)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded trained model: {model_path}")
        else:
            # Use pre-trained YOLOv11n for initialization
            self.model = YOLO('yolo11n.pt')
            logger.warning("Using pre-trained YOLOv11n. Train custom model for better accuracy.")
        
        # Move model to device
        self.model.to(self.device)
        
        # Vietnamese traffic sign classes (7 classes from dataset)
        self.class_names = {
            0: "Cấm con lái",
            1: "Cấm dừng và đỗ", 
            2: "Cấm ngược chiều",
            3: "Cấm rẽ",
            4: "Giới hạn tốc độ",
            5: "Hiệu lệnh",
            6: "Nguy hiểm"
        }
    
    def detect_signs(self, image_path_or_array, return_crops=True):
        """
        Detect traffic signs in image using YOLOv11
        
        Args:
            image_path_or_array: Image file path or numpy array
            return_crops: Whether to return cropped sign images
            
        Returns:
            dict: Detection results with bboxes, confidences, and cropped images
        """
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
            else:
                image = image_path_or_array.copy()
            
            if image is None:
                return {"success": False, "error": "Could not load image"}
            
            # Run YOLOv11 detection
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            cropped_signs = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        bbox = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Create detection dict
                        detection = {
                            'bbox': bbox.tolist(),
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, f"Class_{class_id}")
                        }
                        detections.append(detection)
                        
                        # Extract cropped sign for classification
                        if return_crops:
                            x1, y1, x2, y2 = bbox
                            # Add padding for better classification
                            padding = 10
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding) 
                            x2 = min(image.shape[1], x2 + padding)
                            y2 = min(image.shape[0], y2 + padding)
                            
                            cropped = image[y1:y2, x1:x2]
                            if cropped.size > 0:
                                cropped_signs.append(cropped)
            
            return {
                "success": True,
                "detections": detections,
                "cropped_signs": cropped_signs if return_crops else [],
                "image_shape": image.shape,
                "num_detections": len(detections)
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"success": False, "error": str(e)}

    def detect(self, image, conf_threshold=0.5, iou_threshold=0.45):

        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Không đọc được ảnh. Kiểm tra đường dẫn hoặc định dạng.")

        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.class_names.get(cls, f"class_{cls}")
                })

        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': image.shape
        }

    def visualize(self, image, detections, save_path=None):

        if isinstance(image, str):
            image = cv2.imread(image)

        image_copy = image.copy()

        for det in detections['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['class_name']

            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{label}: {conf:.2f}"
            cv2.putText(image_copy, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, image_copy)

        return image_copy


if __name__ == "__main__":
    detector = YOLODetector()
