import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.yolo_detector import YOLODetector

class TestYOLODetector:
    """Test cases for YOLODetector class"""
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Mock YOLO model"""
        with patch('src.yolo_detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            yield mock_model
    
    @pytest.fixture
    def detector(self, mock_yolo_model):
        """Create detector instance with mocked model"""
        return YOLODetector(model_path="dummy.pt")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        image = np.ones((640, 640, 3), dtype=np.uint8) * 255  # White image
        return image
    
    def test_initialization(self, detector, mock_yolo_model):
        """Test that detector initializes correctly"""
        assert detector.model is not None
        assert isinstance(detector.class_names, dict)
        assert len(detector.class_names) == 15  # Should have 15 classes
    
    def test_detect_with_numpy_array(self, detector, mock_yolo_model, sample_image):
        """Test detection with numpy array input"""
        # Mock the predict method
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [np.array([[100, 100, 200, 200]])]
        mock_result.boxes.conf = [np.array([0.8])]
        mock_result.boxes.cls = [np.array([0])]
        mock_yolo_model.predict.return_value = [mock_result]
        
        results = detector.detect(sample_image)
        
        assert results['num_detections'] == 1
        assert len(results['detections']) == 1
        assert results['detections'][0]['confidence'] == 0.8
        assert results['detections'][0]['class_id'] == 0
    
    def test_detect_with_file_path(self, detector, mock_yolo_model):
        """Test detection with file path input"""
        # Mock cv2.imread
        with patch('src.yolo_detector.cv2.imread') as mock_imread:
            mock_imread.return_value = np.ones((640, 640, 3), dtype=np.uint8) * 255
            
            # Mock the predict method
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = [np.array([[50, 50, 150, 150]])]
            mock_result.boxes.conf = [np.array([0.9])]
            mock_result.boxes.cls = [np.array([1])]
            mock_yolo_model.predict.return_value = [mock_result]
            
            results = detector.detect("test_image.jpg")
            
            mock_imread.assert_called_once_with("test_image.jpg")
            assert results['num_detections'] == 1
            assert results['detections'][0]['confidence'] == 0.9
    
    def test_detect_no_detections(self, detector, mock_yolo_model, sample_image):
        """Test detection when no objects are found"""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = []
        mock_result.boxes.conf = []
        mock_result.boxes.cls = []
        mock_yolo_model.predict.return_value = [mock_result]
        
        results = detector.detect(sample_image)
        
        assert results['num_detections'] == 0
        assert len(results['detections']) == 0
    
    def test_detect_multiple_detections(self, detector, mock_yolo_model, sample_image):
        """Test detection with multiple objects"""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [
            np.array([[100, 100, 200, 200]]),
            np.array([[300, 300, 400, 400]])
        ]
        mock_result.boxes.conf = [np.array([0.8]), np.array([0.7])]
        mock_result.boxes.cls = [np.array([0]), np.array([1])]
        mock_yolo_model.predict.return_value = [mock_result]
        
        results = detector.detect(sample_image)
        
        assert results['num_detections'] == 2
        assert len(results['detections']) == 2
        assert results['detections'][0]['confidence'] == 0.8
        assert results['detections'][1]['confidence'] == 0.7
    
    def test_detect_with_confidence_threshold(self, detector, mock_yolo_model, sample_image):
        """Test detection with custom confidence threshold"""
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [np.array([[100, 100, 200, 200]])]
        mock_result.boxes.conf = [np.array([0.3])]  # Below default threshold
        mock_result.boxes.cls = [np.array([0])]
        mock_yolo_model.predict.return_value = [mock_result]
        
        # Test with lower threshold
        results = detector.detect(sample_image, conf_threshold=0.2)
        
        assert results['num_detections'] == 1
    
    def test_visualize_method(self, detector, mock_yolo_model, sample_image):
        """Test visualize method"""
        detections = {
            'detections': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.8,
                    'class_name': 'speed_limit_20'
                }
            ]
        }
        
        result_image = detector.visualize(sample_image, detections)
        
        assert result_image.shape == sample_image.shape
        assert result_image.dtype == sample_image.dtype
    
    def test_visualize_with_save_path(self, detector, mock_yolo_model, sample_image, tmp_path):
        """Test visualize method with save path"""
        detections = {
            'detections': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.8,
                    'class_name': 'speed_limit_20'
                }
            ]
        }
        
        save_path = tmp_path / "test_output.jpg"
        result_image = detector.visualize(sample_image, detections, save_path=str(save_path))
        
        assert save_path.exists()
        assert result_image.shape == sample_image.shape
    
    def test_invalid_image_path(self, detector):
        """Test detection with invalid image path"""
        with patch('src.yolo_detector.cv2.imread') as mock_imread:
            mock_imread.return_value = None
            
            with pytest.raises(Exception):
                detector.detect("invalid_path.jpg")
    
    def test_config_loading(self):
        """Test that config is loaded correctly"""
        detector = YOLODetector()
        assert hasattr(detector, 'config')
        assert 'classes' in detector.config
        assert 'names' in detector.config['classes']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])