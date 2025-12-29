#!/usr/bin/env python3
"""
Setup script for Traffic Sign Detection Project
This script helps users set up the project structure and download necessary files
"""

import os
from pathlib import Path
import yaml

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        'data/raw/archive',
        'data/processed',
        'models/yolo11',
        'models/cnn',
        'logs',
        'results/images',
        'results/metrics',
        'results/reports',
        'static/css',
        'static/js',
        'static/images',
        'templates',
        'tests',
        'config'  # thêm config để lưu YAML
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create .gitkeep files in empty directories
    for directory in ['models/yolo11', 'models/cnn', 'logs', 'results/images', 
                     'results/metrics', 'results/reports']:
        (Path(directory) / '.gitkeep').touch()
        print(f"Created .gitkeep in: {directory}")

def download_sample_data():
    """Download a small sample dataset for testing"""
    print("Downloading sample dataset...")
    
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README_SAMPLE.md with UTF-8 encoding
    (sample_dir / "README_SAMPLE.md").write_text("""\
🚦 Traffic Sign Detection Sample Dataset

This directory should contain your traffic sign dataset.
The full dataset is too large for Git, so users should download it separately.

Expected structure:
data/raw/archive/
├── images/          # .jpg files
├── labels/          # .txt files (YOLO format)
├── classes.txt      # Class names
├── classes_en.txt   # English class names
└── classes_vie.txt  # Vietnamese class names

You can download datasets from:
- German Traffic Sign Detection Benchmark
- TT100K Dataset
- BelgiumTS Dataset
""", encoding="utf-8")
    
    print("Sample data instructions created")

def create_minimal_configs():
    """Create minimal configuration files"""
    
    # Server config
    server_config = {
        'server': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'reload': False
        },
        'models': {
            'yolo': {
                'path': 'models/yolo11/train/weights/best.pt',
                'confidence': 0.5,
                'iou_threshold': 0.45
            },
            'cnn': {
                'path': 'models/cnn/classifier.h5',
                'input_size': [224, 224]
            }
        },
        'processing': {
            'max_image_size': 5242880,
            'allowed_extensions': ['.jpg', '.jpeg', '.png'],
            'output_format': 'json'
        },
        'cors': {
            'allow_origins': ['*'],
            'allow_methods': ['*'],
            'allow_headers': ['*']
        }
    }
    
    with open('config/server_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(server_config, f, default_flow_style=False)
    print("Created server config")
    
    # YOLO config
    yolo_config = {
        'model': {
            'name': 'yolo11n.pt',
            'pretrained': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 2,
            'img_size': 416,
            'device': 'cpu',
            'workers': 8,
            'patience': 50,
            'optimizer': {
                'name': 'SGD',
                'lr': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0
            }
        },
        'paths': {
            'dataset': 'data/processed',
            'train': 'data/processed/train',
            'val': 'data/processed/val',
            'test': 'data/processed/test',
            'output': 'models/yolo11'
        },
        'classes': {
            'names': {
                0: "speed_limit_20",
                1: "speed_limit_30",
                2: "speed_limit_50",
                3: "speed_limit_60",
                4: "speed_limit_70",
                5: "speed_limit_80",
                6: "no_overtaking",
                7: "no_entry",
                8: "danger",
                9: "mandatory_left",
                10: "mandatory_right",
                11: "mandatory_straight",
                12: "stop",
                13: "yield",
                14: "priority_road"
            }
        }
    }
    
    with open('config/yolo_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    print("Created YOLO config")

def main():
    """Main setup function"""
    print("🚦 Setting up Traffic Sign Detection Project")
    print("=" * 50)
    
    create_project_structure()
    print()
    
    download_sample_data()
    print()
    
    create_minimal_configs()
    print()
    
    print("✅ Project setup completed!")
    print()
    print("Next steps:")
    print("1. Add your dataset to data/raw/archive/")
    print("2. Run: python src/data_preprocessing.py")
    print("3. Run: python src/training/train_yolo.py")
    print("4. Start server: python server/app.py")
    print()
    print("Note: Large dataset files are excluded from Git by .gitignore")

if __name__ == "__main__":
    main()
