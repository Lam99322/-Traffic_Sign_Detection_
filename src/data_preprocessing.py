
"""
Advanced Data Preprocessing Pipeline for Traffic Sign Detection
Features: Blur detection, Image enhancement, Data augmentation, YOLO format processing
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import logging
import matplotlib.pyplot as plt
import random
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path="config/yolo_config.yaml"):
        """Initialize data preprocessor with config"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path("data/raw/archive")
        self.processed_data_path = Path("data/processed")
        
        # Load class mappings
        self.classes = self.load_class_mappings()
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'blurry_images': 0,
            'missing_labels': 0,
            'class_distribution': {},
            'augmented_images': 0
        }
    
    def load_class_mappings(self):
        """Load class mappings from files"""
        classes = {}
        
        # Load basic class names
        classes_file = self.raw_data_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
                classes = {i: name for i, name in enumerate(class_names)}
        
        # Load English names
        classes_en_file = self.raw_data_path / "classes_en.txt"
        if classes_en_file.exists():
            with open(classes_en_file, 'r', encoding='utf-8') as f:
                en_names = [line.strip() for line in f.readlines() if line.strip()]
                for i, name in enumerate(en_names):
                    if i in classes:
                        classes[i] = {'id': classes[i], 'en': name}
        
        # Load Vietnamese names
        classes_vie_file = self.raw_data_path / "classes_vie.txt"
        if classes_vie_file.exists():
            with open(classes_vie_file, 'r', encoding='utf-8') as f:
                vie_names = [line.strip() for line in f.readlines() if line.strip()]
                for i, name in enumerate(vie_names):
                    if i in classes and isinstance(classes[i], dict):
                        classes[i]['vi'] = name
                    elif i in classes:
                        classes[i] = {'id': classes[i], 'vi': name}
        
        logger.info(f"Loaded {len(classes)} classes")
        return classes
    
    def detect_blur(self, image_path, threshold=100.0):
        """
        Detect blurry images using Laplacian variance
        Args:
            image_path: Path to image
            threshold: Blur threshold (lower = more blurry)
        Returns:
            bool: True if image is sharp, False if blurry
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return laplacian_var >= threshold
        except Exception as e:
            logger.error(f"Error checking blur for {image_path}: {e}")
            return False
    
    def enhance_image(self, image):
        """
        Enhance image quality using CLAHE and denoising
        """
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge and convert back
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
            
            # Denoise
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def load_yolo_annotations(self, annotation_path):
        """Load YOLO format annotations from file"""
        annotations = []
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Validate values
                            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                                annotations.append([class_id, x_center, y_center, width, height])
                            else:
                                logger.warning(f"Invalid annotation values in {annotation_path}: {line.strip()}")
                        except ValueError as e:
                            logger.warning(f"Error parsing annotation in {annotation_path}: {line.strip()} - {e}")
        return annotations
    
    def save_yolo_annotations(self, annotations, output_path):
        """Save annotations in YOLO format"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    def augment_image(self, image, annotations, augment_type='random'):
        """
        Apply data augmentation to image and adjust annotations accordingly
        """
        try:
            aug_image = image.copy()
            aug_annotations = annotations.copy()
            
            if augment_type == 'random':
                augment_type = random.choice(['brightness', 'noise', 'blur', 'flip'])
            
            if augment_type == 'brightness':
                # Random brightness adjustment
                hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV)
                brightness = random.uniform(-30, 30)
                hsv[:,:,2] = cv2.add(hsv[:,:,2], int(brightness))
                aug_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            elif augment_type == 'noise':
                # Add gaussian noise
                noise = np.random.randint(0, 25, aug_image.shape, dtype=np.uint8)
                aug_image = cv2.add(aug_image, noise)
            
            elif augment_type == 'blur':
                # Apply slight blur
                kernel_size = random.choice([3, 5])
                aug_image = cv2.GaussianBlur(aug_image, (kernel_size, kernel_size), 0)
            
            elif augment_type == 'flip':
                # Horizontal flip
                aug_image = cv2.flip(aug_image, 1)
                # Adjust annotations for flip
                for i, ann in enumerate(aug_annotations):
                    class_id, x_center, y_center, width, height = ann
                    x_center = 1.0 - x_center  # Flip x coordinate
                    aug_annotations[i] = [class_id, x_center, y_center, width, height]
            
            return aug_image, aug_annotations
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return image, annotations
    
    def validate_sample(self, image_path, label_path, blur_threshold=100):
        """
        Validate a single image-label pair
        Returns: (is_valid, reason)
        """
        try:
            # Check if files exist
            if not image_path.exists():
                return False, "Image file not found"
            
            if not label_path.exists():
                return False, "Label file not found"
            
            # Check image readability
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "Cannot read image file"
            
            # Check image dimensions
            h, w = image.shape[:2]
            if h < 32 or w < 32:
                return False, f"Image too small: {w}x{h}"
            
            # Check blur
            if not self.detect_blur(image_path, blur_threshold):
                return False, "Image is too blurry"
            
            # Check annotations
            annotations = self.load_yolo_annotations(label_path)
            if len(annotations) == 0:
                return False, "No valid annotations found"
            
            # Validate class IDs
            for ann in annotations:
                class_id = ann[0]
                if class_id < 0 or class_id >= len(self.classes):
                    return False, f"Invalid class ID: {class_id}"
            
            return True, "Valid sample"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def split_dataset(self, valid_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train, validation, and test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Stratified split to maintain class distribution
        random.shuffle(valid_samples)
        
        # First split: train and temp (val + test)
        split_idx1 = int(len(valid_samples) * train_ratio)
        train_samples = valid_samples[:split_idx1]
        temp_samples = valid_samples[split_idx1:]
        
        # Second split: val and test
        split_idx2 = int(len(temp_samples) * (val_ratio / (val_ratio + test_ratio)))
        val_samples = temp_samples[:split_idx2]
        test_samples = temp_samples[split_idx2:]
        
        return train_samples, val_samples, test_samples
        
    def process_split(self, samples, split_name, augment_factor=1):
        """
        Process images for a specific split
        Args:
            samples: List of (image_path, label_path) tuples
            split_name: 'train', 'val', or 'test'
            augment_factor: Number of augmented copies to create (only for train)
        """
        logger.info(f"Processing {split_name} split with {len(samples)} samples...")
        
        # Create output directories
        split_dir = self.processed_data_path / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        for img_path, label_path in tqdm(samples, desc=f"Processing {split_name}"):
            try:
                # Load and enhance image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                enhanced_image = self.enhance_image(image)
                
                # Load annotations
                annotations = self.load_yolo_annotations(label_path)
                if len(annotations) == 0:
                    continue
                
                # Save original enhanced image
                output_img_path = images_dir / f"{img_path.stem}.jpg"
                output_label_path = labels_dir / f"{img_path.stem}.txt"
                
                cv2.imwrite(str(output_img_path), enhanced_image)
                self.save_yolo_annotations(annotations, output_label_path)
                
                processed_count += 1
                
                # Update statistics
                for ann in annotations:
                    class_id = ann[0]
                    if class_id not in self.stats['class_distribution']:
                        self.stats['class_distribution'][class_id] = 0
                    self.stats['class_distribution'][class_id] += 1
                
                # Apply augmentations for training data
                if split_name == 'train' and augment_factor > 1:
                    for aug_idx in range(augment_factor - 1):
                        try:
                            aug_image, aug_annotations = self.augment_image(enhanced_image, annotations)
                            
                            # Save augmented image
                            aug_img_name = f"{img_path.stem}_aug_{aug_idx}.jpg"
                            aug_label_name = f"{img_path.stem}_aug_{aug_idx}.txt"
                            
                            aug_img_path = images_dir / aug_img_name
                            aug_label_path = labels_dir / aug_label_name
                            
                            cv2.imwrite(str(aug_img_path), aug_image)
                            self.save_yolo_annotations(aug_annotations, aug_label_path)
                            
                            self.stats['augmented_images'] += 1
                            
                        except Exception as e:
                            logger.warning(f"Augmentation failed for {img_path}: {e}")
                            continue
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"{split_name}: {processed_count} images processed")
        return processed_count
    
    def create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        class_names_dict = {}
        for i, class_info in self.classes.items():
            if isinstance(class_info, dict):
                class_names_dict[i] = class_info.get('id', f'class_{i}')
            else:
                class_names_dict[i] = class_info
        
        data_config = {
            'path': str(self.processed_data_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': class_names_dict
        }
        
        yaml_path = self.processed_data_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Created data.yaml with {len(self.classes)} classes")
        return yaml_path
    
    def generate_statistics(self):
        """Generate and save dataset statistics"""
        stats_extended = self.stats.copy()
        stats_extended['timestamp'] = datetime.now().isoformat()
        stats_extended['class_info'] = self.classes
        
        # Calculate additional stats
        if self.stats['class_distribution']:
            total_annotations = sum(self.stats['class_distribution'].values())
            stats_extended['total_annotations'] = total_annotations
            
            # Class distribution percentages
            class_percentages = {}
            for class_id, count in self.stats['class_distribution'].items():
                class_percentages[class_id] = (count / total_annotations) * 100
            stats_extended['class_percentages'] = class_percentages
        
        # Save statistics
        stats_path = self.processed_data_path / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_extended, f, indent=2, ensure_ascii=False)
        
        # Create visualization
        self.create_visualization()
        
        return stats_extended
    
    def create_visualization(self):
        """Create visualization of dataset statistics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Class distribution
            if self.stats['class_distribution']:
                class_ids = list(self.stats['class_distribution'].keys())
                counts = list(self.stats['class_distribution'].values())
                class_names = [self.classes.get(i, f'Class {i}') for i in class_ids]
                
                axes[0, 0].bar(range(len(class_ids)), counts)
                axes[0, 0].set_title('Class Distribution')
                axes[0, 0].set_xlabel('Class')
                axes[0, 0].set_ylabel('Number of Annotations')
                axes[0, 0].set_xticks(range(len(class_ids)))
                axes[0, 0].set_xticklabels([f'{i}' for i in class_ids], rotation=45)
            
            # 2. Processing statistics
            categories = ['Total', 'Valid', 'Blurry', 'Missing Labels', 'Augmented']
            values = [
                self.stats['total_images'],
                self.stats['valid_images'], 
                self.stats['blurry_images'],
                self.stats['missing_labels'],
                self.stats['augmented_images']
            ]
            
            axes[0, 1].bar(categories, values, color=['blue', 'green', 'red', 'orange', 'purple'])
            axes[0, 1].set_title('Processing Statistics')
            axes[0, 1].set_ylabel('Number of Images')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # 3. Data splits
            splits = ['Train', 'Val', 'Test']
            split_counts = []
            for split in ['train', 'val', 'test']:
                split_dir = self.processed_data_path / split / 'images'
                if split_dir.exists():
                    count = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
                    split_counts.append(count)
                else:
                    split_counts.append(0)
            
            axes[1, 0].pie(split_counts, labels=splits, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Dataset Splits')
            
            # 4. Success rate
            total = self.stats['total_images'] if self.stats['total_images'] > 0 else 1
            success_rate = (self.stats['valid_images'] / total) * 100
            failure_rate = 100 - success_rate
            
            axes[1, 1].pie([success_rate, failure_rate], 
                          labels=['Valid', 'Invalid'], 
                          autopct='%1.1f%%', 
                          colors=['green', 'red'],
                          startangle=90)
            axes[1, 1].set_title('Processing Success Rate')
            
            plt.tight_layout()
            viz_path = self.processed_data_path / "dataset_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {viz_path}")
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
    
    def prepare_yolo_dataset(self, blur_threshold=100, augment_factor=3, 
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Main preprocessing pipeline
        """
        logger.info("🚀 Starting dataset preprocessing...")
        
        # Check raw data structure
        if not self.raw_data_path.exists():
            logger.error(f"❌ Raw data path not found: {self.raw_data_path}")
            return False
        
        images_path = self.raw_data_path / "images"
        labels_path = self.raw_data_path / "labels"
        
        if not images_path.exists() or not labels_path.exists():
            logger.error("❌ Images or labels directory not found!")
            logger.info("📁 Expected structure:")
            logger.info("   data/raw/archive/")
            logger.info("   ├── images/")
            logger.info("   ├── labels/")
            logger.info("   ├── classes.txt")
            logger.info("   ├── classes_en.txt")
            logger.info("   └── classes_vie.txt")
            return False
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(images_path.glob(ext)))
            image_files.extend(list(images_path.glob(ext.upper())))
        
        self.stats['total_images'] = len(image_files)
        logger.info(f"📁 Found {len(image_files)} images")
        
        if len(image_files) == 0:
            logger.error("❌ No images found in dataset")
            return False
        
        # Validate samples
        logger.info("🔍 Validating samples...")
        valid_samples = []
        
        for img_path in tqdm(image_files, desc="Validating"):
            label_path = labels_path / f"{img_path.stem}.txt"
            
            is_valid, reason = self.validate_sample(img_path, label_path, blur_threshold)
            
            if is_valid:
                valid_samples.append((img_path, label_path))
                self.stats['valid_images'] += 1
            else:
                if "blurry" in reason.lower():
                    self.stats['blurry_images'] += 1
                elif "not found" in reason.lower():
                    self.stats['missing_labels'] += 1
        
        logger.info(f"✅ Valid samples: {len(valid_samples)}")
        logger.info(f"❌ Invalid samples: {len(image_files) - len(valid_samples)}")
        
        if len(valid_samples) == 0:
            logger.error("❌ No valid samples found!")
            return False
        
        # Split dataset
        train_samples, val_samples, test_samples = self.split_dataset(
            valid_samples, train_ratio, val_ratio, test_ratio
        )
        
        logger.info("📊 Dataset splits:")
        logger.info(f"   Train: {len(train_samples)} samples")
        logger.info(f"   Val: {len(val_samples)} samples") 
        logger.info(f"   Test: {len(test_samples)} samples")
        
        # Process splits
        splits_data = [
            (train_samples, 'train', augment_factor),
            (val_samples, 'val', 1),
            (test_samples, 'test', 1)
        ]
        
        total_processed = 0
        for samples, split_name, aug_factor in splits_data:
            processed = self.process_split(samples, split_name, aug_factor)
            total_processed += processed
        
        # Create data.yaml
        self.create_data_yaml()
        
        # Generate statistics
        final_stats = self.generate_statistics()
        
        # Summary
        logger.info("=" * 60)
        logger.info("📋 PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images found: {self.stats['total_images']}")
        logger.info(f"Valid images: {self.stats['valid_images']}")
        logger.info(f"Blurry images filtered: {self.stats['blurry_images']}")
        logger.info(f"Missing labels: {self.stats['missing_labels']}")
        logger.info(f"Augmented images created: {self.stats['augmented_images']}")
        logger.info(f"Success rate: {(self.stats['valid_images']/self.stats['total_images']*100):.1f}%")
        logger.info(f"Output directory: {self.processed_data_path}")
        logger.info("=" * 60)
        
        return True

def main():
    """Main entry point for data preprocessing"""
    logger.info("🚦 Traffic Sign Dataset Preprocessor")
    logger.info("=" * 50)
    
    try:
        preprocessor = DataPreprocessor()
        
        # Run preprocessing with custom parameters
        success = preprocessor.prepare_yolo_dataset(
            blur_threshold=100,    # Adjust for blur sensitivity
            augment_factor=3,      # Number of augmented copies per training image
            train_ratio=0.7,       # 70% for training
            val_ratio=0.15,        # 15% for validation  
            test_ratio=0.15        # 15% for testing
        )
        
        if success:
            print("\n✅ Dataset preprocessing completed successfully!")
            print(f"📁 Processed data available in: {preprocessor.processed_data_path}")
            print("📋 Files created:")
            print("   - data.yaml (YOLO config)")
            print("   - statistics.json (dataset stats)")
            print("   - dataset_visualization.png (charts)")
            print("\n🚀 Next steps:")
            print("1. Train YOLO model: python src/training/train_yolo.py")
            print("2. Train CNN classifier: python src/training/train_cnn.py") 
            print("3. Run web server: python server/app_demo.py")
        else:
            print("\n❌ Preprocessing failed. Check logs for details.")
            print("💡 Make sure dataset is in: data/raw/archive/")
            
    except Exception as e:
        logger.error(f"❌ Preprocessing failed with error: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()