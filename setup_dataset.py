#!/usr/bin/env python3
"""
Setup script to configure the existing dataset for training
Dataset: Vietnamese Traffic Signs with 7 classes
"""
import os
import yaml
import json
import shutil
from pathlib import Path

def setup_dataset():
    """Setup the existing dataset for training"""
    print("🚦 Setting up Vietnamese Traffic Signs Dataset")
    print("=" * 60)
    
    # Define paths
    base_path = Path(".")
    data_path = base_path / "data"
    
    # Check dataset structure
    print("📊 Dataset Overview:")
    print("=" * 30)
    
    # Count images
    train_images = len(list((data_path / "train" / "images").glob("*"))) if (data_path / "train" / "images").exists() else 0
    valid_images = len(list((data_path / "valid" / "images").glob("*"))) if (data_path / "valid" / "images").exists() else 0
    test_images = len(list((data_path / "test" / "images").glob("*"))) if (data_path / "test" / "images").exists() else 0
    
    print(f"📁 Training images: {train_images}")
    print(f"📁 Validation images: {valid_images}")
    print(f"📁 Test images: {test_images}")
    print(f"📁 Total images: {train_images + valid_images + test_images}")
    
    # Read dataset config
    data_yaml_path = data_path / "data.yaml"
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"\n🏷️  Classes ({dataset_config['nc']}):")
        for i, class_name in enumerate(dataset_config['names']):
            print(f"  {i}: {class_name}")
    
    # Create updated config for training
    print(f"\n⚙️  Creating training configuration...")
    
    # Update data.yaml with absolute paths
    updated_config = {
        'path': str(data_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images', 
        'test': 'test/images',
        'nc': 7,
        'names': [
            'Cam con lai',      # 0: No passing
            'Cam dung va do',   # 1: No stopping and parking
            'Cam nguoc chieu',  # 2: No entry
            'Cam re',          # 3: No turn
            'Gioi han toc do', # 4: Speed limit
            'Hieu lenh',       # 5: Mandatory
            'Nguy hiem'        # 6: Warning/Danger
        ]
    }
    
    # Write updated config
    config_path = data_path / "dataset_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(updated_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Dataset config saved: {config_path}")
    
    # Create class mapping for Vietnamese TTS
    class_mapping = {
        0: {
            'en': 'No passing', 
            'vi': 'Cấm con lái',
            'description': 'Biển báo cấm các xe cơ giới vượt'
        },
        1: {
            'en': 'No stopping and parking',
            'vi': 'Cấm dừng và đỗ', 
            'description': 'Biển báo cấm dừng xe và đỗ xe'
        },
        2: {
            'en': 'No entry',
            'vi': 'Cấm ngược chiều',
            'description': 'Biển báo cấm đi ngược chiều'
        },
        3: {
            'en': 'No turn',
            'vi': 'Cấm rẽ',
            'description': 'Biển báo cấm rẽ trái hoặc rẽ phải'
        },
        4: {
            'en': 'Speed limit',
            'vi': 'Giới hạn tốc độ',
            'description': 'Biển báo giới hạn tốc độ tối đa'
        },
        5: {
            'en': 'Mandatory',
            'vi': 'Hiệu lệnh',
            'description': 'Biển báo hiệu lệnh bắt buộc'
        },
        6: {
            'en': 'Warning',
            'vi': 'Nguy hiểm',
            'description': 'Biển báo cảnh báo nguy hiểm'
        }
    }
    
    # Save class mapping
    mapping_path = data_path / "class_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Class mapping saved: {mapping_path}")
    
    # Check label files
    print(f"\n📝 Checking label files...")
    train_labels = len(list((data_path / "train" / "labels").glob("*.txt"))) if (data_path / "train" / "labels").exists() else 0
    valid_labels = len(list((data_path / "valid" / "labels").glob("*.txt"))) if (data_path / "valid" / "labels").exists() else 0
    test_labels = len(list((data_path / "test" / "labels").glob("*.txt"))) if (data_path / "test" / "labels").exists() else 0
    
    print(f"📄 Training labels: {train_labels}")
    print(f"📄 Validation labels: {valid_labels}")
    print(f"📄 Test labels: {test_labels}")
    
    # Create training script for this dataset
    create_training_script(data_path, updated_config)
    
    print(f"\n🎉 Dataset setup complete!")
    print(f"📈 Ready for training with {train_images + valid_images + test_images} total images")
    print(f"🏷️  7 Vietnamese traffic sign classes")
    
    return updated_config

def create_training_script(data_path, config):
    """Create a specific training script for this dataset"""
    
    training_script = f'''#!/usr/bin/env python3
"""
Training script for Vietnamese Traffic Signs Dataset
Uses the prepared dataset in data/ folder
"""
import os
from ultralytics import YOLO
import torch
import yaml

def train_vietnamese_traffic_signs():
    """Train YOLOv11 on Vietnamese traffic signs"""
    print("🚦 Training Vietnamese Traffic Signs Detection Model")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {{device}}")
    
    # Load dataset config
    dataset_config = "{data_path / 'dataset_config.yaml'}"
    
    # Initialize model
    model = YOLO('yolo11n.pt')  # Start with nano model
    print("✅ Model loaded: YOLOv11 Nano")
    
    # Training parameters
    results = model.train(
        data=dataset_config,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='runs/traffic_signs',
        name='vietnamese_signs_v1',
        
        # Optimization
        optimizer='AdamW',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        
        # Validation
        val=True,
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        plots=True,
        
        # Performance
        workers=4,
        verbose=True
    )
    
    print("\\n📊 Training Results:")
    print(f"📈 mAP50: {{results.results_dict.get('metrics/mAP50(B)', 'N/A')}}")
    print(f"📈 mAP50-95: {{results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}}")
    
    # Export model
    print("\\n📦 Exporting model...")
    model.export(format='onnx')
    print("✅ Model exported to ONNX format")
    
    return model, results

if __name__ == "__main__":
    model, results = train_vietnamese_traffic_signs()
'''
    
    script_path = Path("train_vietnamese_signs.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    print(f"✅ Training script created: {script_path}")

def validate_dataset():
    """Validate the dataset structure and content"""
    print("\n🔍 Validating dataset...")
    
    data_path = Path("data")
    
    # Check required directories
    required_dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels", 
        "test/images", "test/labels"
    ]
    
    for dir_path in required_dirs:
        full_path = data_path / dir_path
        if full_path.exists():
            count = len(list(full_path.glob("*")))
            print(f"✅ {dir_path}: {count} files")
        else:
            print(f"❌ {dir_path}: Missing!")
    
    # Sample a few label files to validate format
    label_dir = data_path / "train" / "labels"
    if label_dir.exists():
        label_files = list(label_dir.glob("*.txt"))[:3]
        print(f"\\n📄 Sample label files:")
        for label_file in label_files:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                lines = content.split('\\n') if content else []
                print(f"  {label_file.name}: {len(lines)} objects")

if __name__ == "__main__":
    # Setup dataset
    config = setup_dataset()
    
    # Validate
    validate_dataset()
    
    print(f"\\n🚀 Next steps:")
    print(f"1. Run: python train_vietnamese_signs.py")
    print(f"2. Monitor training in runs/traffic_signs/")
    print(f"3. Use trained model in web interface")
    print(f"\\n📊 Dataset ready with {config['nc']} classes!")