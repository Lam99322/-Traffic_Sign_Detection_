
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import yaml
import sys
sys.path.append('.')
from src.cnn_classifier import CNNClassifier

class CNNTrainer:
    def __init__(self, config_path="config/yolo_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_classes = len(self.config['classes']['names'])
        self.classifier = CNNClassifier(self.num_classes)
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32, img_size=224):
        """Tạo data generators cho training"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            # class_mode='categorical'
            class_mode='sparse',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(128, 128),
            batch_size=32,
            # class_mode='categorical' 
            class_mode='sparse',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def get_callbacks(self, model_save_path="models/cnn"):
        """Tạo callbacks cho training"""
        callbacks = [
            ModelCheckpoint(
                f"{model_save_path}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=f"{model_save_path}/logs",
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, train_dir, val_dir, epochs=50, batch_size=32):
        """Train CNN model"""
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators(
            train_dir, val_dir, batch_size
        )
        
        print("Building model...")
        self.classifier.build_model()
        self.classifier.model.summary()
        
        print("Starting training...")
        callbacks = self.get_callbacks()
        
        history = self.classifier.train(
            train_gen,
            val_gen,
            epochs=epochs,
            callbacks=callbacks
        )
        
        print("Training completed!")
        return history

def main():
    """Main entry point for CNN training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN model for traffic sign classification')
    parser.add_argument('--config', '-c', default='config/yolo_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--train-dir', '-t', default='data/processed/train',
                       help='Path to training data directory')
    parser.add_argument('--val-dir', '-v', default='data/processed/val',
                       help='Path to validation data directory')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--output-dir', '-o', default='models/cnn',
                       help='Output directory for saved models')
    
    args = parser.parse_args()
    
    try:
        trainer = CNNTrainer(args.config)
        
        print(f"Training directory: {args.train_dir}")
        print(f"Validation directory: {args.val_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        
        # Train model
        history = trainer.train(
            args.train_dir,
            args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save final model
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        final_model_path = os.path.join(args.output_dir, 'final_model.h5')
        trainer.classifier.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Plot training history
        from src.utils.visualization import Visualizer
        visualizer = Visualizer()
        history_plot_path = os.path.join(args.output_dir, 'training_history.png')
        visualizer.plot_training_history(history, save_path=history_plot_path)
        print(f"Training history plot saved to: {history_plot_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()