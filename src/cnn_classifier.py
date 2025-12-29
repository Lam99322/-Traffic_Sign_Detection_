import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class TrafficSignCNN:
    """
    CNN Traffic Sign Classifier using TensorFlow/Keras 2.x
    Classifies cropped traffic sign images from YOLO detection
    """
    
    def __init__(self, model_path=None, input_size=(224, 224)):
        self.input_size = input_size
        self.num_classes = 7  # Vietnamese traffic signs
        
        # Class mapping
        self.class_names = {
            0: "Cấm con lái",
            1: "Cấm dừng và đỗ", 
            2: "Cấm ngược chiều",
            3: "Cấm rẽ",
            4: "Giới hạn tốc độ",
            5: "Hiệu lệnh",
            6: "Nguy hiểm"
        }
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self.model = keras.models.load_model(model_path)
            logger.info(f"Loaded CNN model: {model_path}")
        else:
            self.model = self._build_cnn_model()
            logger.warning("Created new CNN model. Train for better accuracy.")
    
    def _build_cnn_model(self):
        """
        Build CNN architecture for traffic sign classification
        """
        model = keras.Sequential([
            # Input layer
            keras.Input(shape=(*self.input_size, 3)),
            
            # Preprocessing
            layers.Rescaling(1./255),
            
            # Convolutional blocks
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(), 
            layers.Dropout(0.25),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Classification head
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess image for CNN classification
        """
        # Resize to input size
        image_resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_rgb, axis=0)
        
        return image_batch
    
    def classify_signs(self, cropped_images: List[np.ndarray]):
        """
        Classify multiple cropped traffic sign images
        
        Args:
            cropped_images: List of cropped sign images from YOLO
            
        Returns:
            List[Dict]: Classification results for each image
        """
        if not cropped_images:
            return []
        
        results = []
        
        for i, image in enumerate(cropped_images):
            try:
                # Preprocess
                processed_image = self.preprocess_image(image)
                
                # Predict
                predictions = self.model.predict(processed_image, verbose=0)
                
                # Get top prediction
                class_id = np.argmax(predictions[0])
                confidence = float(predictions[0][class_id])
                
                result = {
                    'image_id': i,
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id],
                    'confidence': confidence,
                    'all_predictions': predictions[0].tolist()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Classification error for image {i}: {e}")
                results.append({
                    'image_id': i,
                    'error': str(e)
                })
        
        return results

class VisionTransformerClassifier:
    """
    Vision Transformer (ViT) for Traffic Sign Classification
    More advanced alternative to CNN
    """
    
    def __init__(self, model_path=None, input_size=(224, 224)):
        self.input_size = input_size
        self.num_classes = 7
        self.patch_size = 16
        
        # Class mapping
        self.class_names = {
            0: "Cấm con lái",
            1: "Cấm dừng và đỗ", 
            2: "Cấm ngược chiều",
            3: "Cấm rẽ",
            4: "Giới hạn tốc độ",
            5: "Hiệu lệnh",
            6: "Nguy hiểm"
        }
        
        # Load or create ViT model
        if model_path and Path(model_path).exists():
            self.model = keras.models.load_model(model_path)
            logger.info(f"Loaded ViT model: {model_path}")
        else:
            self.model = self._build_vit_model()
            logger.warning("Created new ViT model. Train for better accuracy.")
    
    def _build_vit_model(self):
        """
        Build Vision Transformer architecture
        """
        # Input
        inputs = keras.Input(shape=(*self.input_size, 3))
        
        # Patch embedding
        patch_dim = self.patch_size * self.patch_size * 3
        num_patches = (self.input_size[0] // self.patch_size) * (self.input_size[1] // self.patch_size)
        
        # Extract patches
        patches = layers.Conv2D(
            filters=patch_dim, 
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid'
        )(inputs)
        
        patches = layers.Reshape((num_patches, patch_dim))(patches)
        
        # Linear projection
        embedding_dim = 256
        patches = layers.Dense(embedding_dim)(patches)
        
        # Add positional embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)
        patches = patches + pos_embedding
        
        # Transformer blocks
        for _ in range(4):  # 4 transformer layers
            # Multi-head attention
            x1 = layers.LayerNormalization()(patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=embedding_dim // 8
            )(x1, x1)
            x2 = layers.Add()([attention_output, patches])
            
            # MLP
            x3 = layers.LayerNormalization()(x2)
            x3 = layers.Dense(embedding_dim * 2, activation='gelu')(x3)
            x3 = layers.Dense(embedding_dim)(x3)
            patches = layers.Add()([x3, x2])
        
        # Classification head
        representation = layers.LayerNormalization()(patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        representation = layers.Dropout(0.3)(representation)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(representation)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def classify_signs(self, cropped_images: List[np.ndarray]):
        """
        Classify using Vision Transformer
        Same interface as CNN classifier
        """
        if not cropped_images:
            return []
        
        results = []
        
        for i, image in enumerate(cropped_images):
            try:
                # Preprocess (same as CNN)
                image_resized = cv2.resize(image, self.input_size)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image_batch = np.expand_dims(image_rgb / 255.0, axis=0)
                
                # Predict with ViT
                predictions = self.model.predict(image_batch, verbose=0)
                
                # Get results
                class_id = np.argmax(predictions[0])
                confidence = float(predictions[0][class_id])
                
                result = {
                    'image_id': i,
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id],
                    'confidence': confidence,
                    'model_type': 'ViT',
                    'all_predictions': predictions[0].tolist()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"ViT classification error for image {i}: {e}")
                results.append({
                    'image_id': i,
                    'error': str(e)
                })
        
        return results
    
    def build_model(self):
        """Xây dựng mô hình CNN"""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Chọn loss phù hợp
        loss_fn = 'sparse_categorical_crossentropy' if self.use_sparse_labels else 'categorical_crossentropy'
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_generator, val_generator, epochs=50, callbacks=None):
        """Train model"""
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        """Dự đoán cho một ảnh"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize và normalize
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        return {
            'class_id': int(class_id),
            'confidence': float(confidence),
            'all_probabilities': predictions[0].tolist()
        }
    
    def save(self, path):
        """Lưu model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Ví dụ: nếu dataset có 2 lớp và nhãn là số nguyên (0 hoặc 1)
    classifier = CNNClassifier(num_classes=2, use_sparse_labels=True)
    model = classifier.build_model()
    model.summary()