"""
Training script for hybrid YOLOv11 + CNN/ViT traffic sign detection system
Supports training both models on Vietnamese traffic signs dataset
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
import yaml
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridModelTrainer:
    """
    Trainer for hybrid YOLOv11 + CNN/ViT system
    """
    
    def __init__(self, config_path="config/yolo_config.yaml"):
        """Initialize trainer with config"""
        self.config = self.load_config(config_path)
        self.data_path = Path("data")
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Vietnamese traffic signs classes
        self.class_names = [
            "Cấm con lái", "Cấm dừng và đỗ", "Cấm ngược chiều",
            "Cấm rẽ", "Giới hạn tốc độ", "Hiệu lệnh", "Nguy hiểm"
        ]
        
    def load_config(self, config_path):
        """Load training configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Default config
            return {
                'yolo': {
                    'epochs': 100,
                    'batch_size': 16,
                    'img_size': 640,
                    'device': 'auto'
                },
                'cnn': {
                    'epochs': 50,
                    'batch_size': 32,
                    'img_size': [224, 224],
                    'learning_rate': 1e-4
                },
                'vit': {
                    'epochs': 50,
                    'batch_size': 16,
                    'img_size': [224, 224],
                    'learning_rate': 1e-5
                }
            }
    
    def train_yolo_detector(self):
        """
        Train YOLOv11 for traffic sign detection
        """
        logger.info("Starting YOLOv11 training for traffic sign detection...")
        
        try:
            # Initialize YOLOv11 model
            model = YOLO('yolov8n.pt')  # Start with pretrained weights
            
            # Prepare dataset config
            dataset_config = {
                'train': str(self.data_path / 'train'),
                'val': str(self.data_path / 'valid'),
                'test': str(self.data_path / 'test'),
                'nc': len(self.class_names),  # Number of classes
                'names': self.class_names
            }
            
            # Save dataset config
            config_path = self.data_path / "dataset.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset_config, f)
            
            # Train the model
            results = model.train(
                data=str(config_path),
                epochs=self.config['yolo']['epochs'],
                batch=self.config['yolo']['batch_size'],
                imgsz=self.config['yolo']['img_size'],
                device=self.config['yolo']['device'],
                name='traffic_signs_yolo',
                project='runs/detect'
            )
            
            # Save trained model
            model_save_path = self.models_path / 'yolo_traffic_signs.pt'
            model.save(str(model_save_path))
            
            logger.info(f"YOLOv11 training completed. Model saved to: {model_save_path}")
            return str(model_save_path)
            
        except Exception as e:
            logger.error(f"YOLOv11 training failed: {e}")
            return None
    
    def prepare_classification_data(self):
        """
        Prepare data for CNN/ViT classification training
        """
        logger.info("Preparing classification dataset...")
        
        # This would typically involve:
        # 1. Loading YOLO detection results
        # 2. Cropping detected signs
        # 3. Creating classification dataset
        
        # For now, use the original dataset structure
        train_path = self.data_path / 'train'
        valid_path = self.data_path / 'valid'
        
        return str(train_path), str(valid_path)
    
    def create_cnn_model(self, input_shape=(224, 224, 3), num_classes=7):
        """
        Create CNN model for traffic sign classification
        """
        model = keras.Sequential([
            keras.Input(shape=input_shape),
            
            # Preprocessing
            keras.layers.Rescaling(1./255),
            
            # Feature extraction
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(256, 3, activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            
            # Classification head
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['cnn']['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn_classifier(self):
        """
        Train CNN classifier for traffic signs
        """
        logger.info("Starting CNN classifier training...")
        
        try:
            # Prepare data
            train_path, valid_path = self.prepare_classification_data()
            
            # Create data generators
            img_size = tuple(self.config['cnn']['img_size'])
            batch_size = self.config['cnn']['batch_size']
            
            train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2
            )
            
            valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            
            # Load data
            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='sparse'
            )
            
            valid_generator = valid_datagen.flow_from_directory(
                valid_path,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='sparse'
            )
            
            # Create and train model
            model = self.create_cnn_model(input_shape=(*img_size, 3))
            
            # Callbacks
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    str(self.models_path / 'cnn_traffic_signs_best.h5'),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                keras.callbacks.EarlyStopping(
                    patience=10,
                    monitor='val_accuracy',
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    monitor='val_accuracy'
                )
            ]
            
            # Train
            history = model.fit(
                train_generator,
                epochs=self.config['cnn']['epochs'],
                validation_data=valid_generator,
                callbacks=callbacks
            )
            
            # Save final model
            model_save_path = self.models_path / 'cnn_traffic_signs.h5'
            model.save(str(model_save_path))
            
            logger.info(f"CNN training completed. Model saved to: {model_save_path}")
            
            # Plot training history
            self.plot_training_history(history, 'CNN')
            
            return str(model_save_path)
            
        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            return None
    
    def create_vit_model(self, input_shape=(224, 224, 3), num_classes=7):
        """
        Create Vision Transformer model
        """
        from tensorflow.keras import layers
        
        patch_size = 16
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        patch_dim = patch_size * patch_size * input_shape[2]
        embedding_dim = 256
        
        inputs = keras.Input(shape=input_shape)
        
        # Patch embedding
        patches = layers.Conv2D(
            filters=patch_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid'
        )(inputs)
        
        patches = layers.Reshape((num_patches, patch_dim))(patches)
        patches = layers.Dense(embedding_dim)(patches)
        
        # Positional embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)
        patches = patches + pos_embedding
        
        # Transformer blocks
        for _ in range(4):
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
        outputs = layers.Dense(num_classes, activation='softmax')(representation)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['vit']['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_history_{model_name.lower()}.png')
        plt.show()
    
    def train_all_models(self):
        """
        Train all models in the hybrid system
        """
        logger.info("Starting training for hybrid traffic sign detection system...")
        
        results = {}
        
        # 1. Train YOLO detector
        logger.info("\n=== TRAINING YOLO DETECTOR ===")
        yolo_path = self.train_yolo_detector()
        results['yolo'] = yolo_path
        
        # 2. Train CNN classifier
        logger.info("\n=== TRAINING CNN CLASSIFIER ===")
        cnn_path = self.train_cnn_classifier()
        results['cnn'] = cnn_path
        
        # 3. Optional: Train ViT classifier
        # vit_path = self.train_vit_classifier()
        # results['vit'] = vit_path
        
        logger.info("\n=== TRAINING COMPLETED ===")
        logger.info(f"Results: {results}")
        
        return results

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train hybrid traffic sign detection models")
    parser.add_argument("--model", choices=['yolo', 'cnn', 'vit', 'all'], default='all',
                       help="Which model to train")
    parser.add_argument("--config", default="config/yolo_config.yaml",
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = HybridModelTrainer(args.config)
    
    # Train specified model(s)
    if args.model == 'yolo':
        trainer.train_yolo_detector()
    elif args.model == 'cnn':
        trainer.train_cnn_classifier()
    elif args.model == 'vit':
        # trainer.train_vit_classifier()
        print("ViT training not implemented yet")
    else:
        trainer.train_all_models()

if __name__ == "__main__":
    main()