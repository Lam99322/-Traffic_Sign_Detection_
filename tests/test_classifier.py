import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cnn_classifier import CNNClassifier

class TestCNNClassifier:
    """Test cases for CNNClassifier class"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return CNNClassifier(num_classes=15)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_model(self):
        """Mock TensorFlow model"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.2, 0.7]])  # Mock prediction
        return mock_model
    
    def test_initialization(self, classifier):
        """Test that classifier initializes correctly"""
        assert classifier.num_classes == 15
        assert classifier.input_shape == (224, 224, 3)
        assert classifier.model is None
    
    def test_build_model(self, classifier):
        """Test model building"""
        model = classifier.build_model()
        
        assert model is not None
        assert isinstance(model, tf.keras.Sequential)
        assert len(model.layers) > 0
        
        # Check output layer
        output_layer = model.layers[-1]
        assert output_layer.units == 15  # Should match num_classes
        assert output_layer.activation.__name__ == 'softmax'
    
    def test_build_model_architecture(self, classifier):
        """Test model architecture details"""
        model = classifier.build_model()
        
        # Check that model has expected layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        
        assert 'Conv2D' in layer_types
        assert 'MaxPooling2D' in layer_types
        assert 'Flatten' in layer_types
        assert 'Dense' in layer_types
        assert 'Dropout' in layer_types
    
    def test_predict_with_numpy_array(self, classifier, sample_image, mock_model):
        """Test prediction with numpy array input"""
        classifier.model = mock_model
        
        result = classifier.predict(sample_image)
        
        assert 'class_id' in result
        assert 'confidence' in result
        assert 'all_probabilities' in result
        assert isinstance(result['class_id'], int)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['all_probabilities'], list)
        
        # Mock prediction should return class 2 with 0.7 confidence
        assert result['class_id'] == 2
        assert result['confidence'] == 0.7
    
    def test_predict_with_file_path(self, classifier, mock_model):
        """Test prediction with file path input"""
        classifier.model = mock_model
        
        # Mock cv2 functions
        with patch('src.cnn_classifier.cv2.imread') as mock_imread, \
             patch('src.cnn_classifier.cv2.cvtColor') as mock_cvtColor:
            
            mock_imread.return_value = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            mock_cvtColor.return_value = mock_imread.return_value
            
            result = classifier.predict("test_image.jpg")
            
            mock_imread.assert_called_once_with("test_image.jpg")
            mock_cvtColor.assert_called_once()
            
            assert result['class_id'] == 2
            assert result['confidence'] == 0.7
    
    def test_predict_preprocessing(self, classifier, sample_image, mock_model):
        """Test that image is preprocessed correctly"""
        classifier.model = mock_model
        
        result = classifier.predict(sample_image)
        
        # Check that model.predict was called
        mock_model.predict.assert_called_once()
        
        # Get the preprocessed image that was passed to predict
        call_args = mock_model.predict.call_args
        preprocessed_image = call_args[0][0]  # First argument, first batch
        
        # Check preprocessing
        assert preprocessed_image.shape == (1, 224, 224, 3)  # Batch size 1
        assert preprocessed_image.dtype == np.float32
        assert np.max(preprocessed_image) <= 1.0  # Should be normalized
        assert np.min(preprocessed_image) >= 0.0
    
    def test_predict_empty_image(self, classifier, mock_model):
        """Test prediction with empty image"""
        classifier.model = mock_model
        
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        
        with pytest.raises(Exception):
            classifier.predict(empty_image)
    
    def test_save_and_load(self, classifier, tmp_path):
        """Test model saving and loading"""
        # Build model first
        classifier.build_model()
        
        save_path = tmp_path / "test_model.h5"
        
        # Mock the save method to avoid actual file operations in tests
        with patch.object(classifier.model, 'save') as mock_save:
            classifier.save(str(save_path))
            mock_save.assert_called_once_with(str(save_path))
    
    def test_load_model(self, classifier, tmp_path):
        """Test model loading"""
        # Mock keras.models.load_model
        with patch('src.cnn_classifier.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            classifier.load("dummy_path.h5")
            
            mock_load.assert_called_once_with("dummy_path.h5")
            assert classifier.model == mock_model
    
    def test_train_method(self, classifier):
        """Test training method"""
        # Mock data generators
        mock_train_gen = Mock()
        mock_val_gen = Mock()
        
        # Mock model methods
        mock_model = Mock()
        mock_model.fit.return_value = Mock()  # Mock history
        classifier.model = mock_model
        
        # Mock callbacks
        mock_callbacks = [Mock()]
        
        with patch.object(classifier, 'get_callbacks', return_value=mock_callbacks):
            history = classifier.train(mock_train_gen, mock_val_gen, epochs=2)
            
            mock_model.fit.assert_called_once_with(
                mock_train_gen,
                validation_data=mock_val_gen,
                epochs=2,
                callbacks=mock_callbacks
            )
            assert history is not None
    
    def test_get_callbacks(self, classifier):
        """Test callback creation"""
        callbacks = classifier.get_callbacks()
        
        assert len(callbacks) == 4  # Should have 4 callbacks
        callback_types = [type(cb).__name__ for cb in callbacks]
        
        assert 'ModelCheckpoint' in callback_types
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types
        assert 'TensorBoard' in callback_types
    
    def test_predict_batch_processing(self, classifier, mock_model):
        """Test batch prediction processing"""
        classifier.model = mock_model
        
        # Create batch of images
        batch_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ]
        
        results = []
        for image in batch_images:
            result = classifier.predict(image)
            results.append(result)
        
        assert len(results) == 2
        assert all('class_id' in result for result in results)
        assert all('confidence' in result for result in results)
    
    def test_model_summary(self, classifier):
        """Test model summary method"""
        classifier.build_model()
        
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            classifier.model.summary()
            mock_print.assert_called()
    
    def test_invalid_model_path(self, classifier):
        """Test loading from invalid path"""
        with patch('src.cnn_classifier.keras.models.load_model') as mock_load:
            mock_load.side_effect = Exception("File not found")
            
            with pytest.raises(Exception):
                classifier.load("invalid_path.h5")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])