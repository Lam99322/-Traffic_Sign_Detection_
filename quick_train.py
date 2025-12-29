#!/usr/bin/env python3
"""
Quick Training Script for Vietnamese Traffic Signs
Uses the existing dataset in data/ folder
"""
import os
import sys
import torch
from pathlib import Path
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    print("🔧 Checking requirements...")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
        print("✅ Ultralytics installed")
    
    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        print("❌ PyYAML not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
        print("✅ PyYAML installed")

def start_training():
    """Start training with the Vietnamese dataset"""
    print("🚦 Starting Vietnamese Traffic Signs Training")
    print("=" * 60)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Check dataset
    dataset_config = Path("data/dataset_config.yaml")
    if not dataset_config.exists():
        print("❌ Dataset config not found. Run setup_dataset.py first!")
        return False
    
    print(f"📊 Dataset config: {dataset_config}")
    
    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ YOLO import failed. Installing ultralytics...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO
    
    # Start training
    print("🎯 Initializing YOLOv11 training...")
    
    # Choose model size based on device
    if device == 'cuda':
        model_size = 'yolo11s.pt'  # Small model for GPU
        batch_size = 16
        epochs = 50
    else:
        model_size = 'yolo11n.pt'  # Nano model for CPU
        batch_size = 4
        epochs = 20
        
    print(f"📦 Loading {model_size}")
    model = YOLO(model_size)
    
    # Training parameters
    print("🏃 Starting training...")
    results = model.train(
        data=str(dataset_config),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        project='runs/traffic_signs',
        name=f'vietnamese_signs_{device}',
        
        # Basic optimization
        patience=10,
        save=True,
        plots=True,
        val=True,
        
        # Performance settings
        workers=2 if device == 'cpu' else 4,
        verbose=True,
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5
    )
    
    print("\n🎉 Training completed!")
    print(f"📊 Results saved in: runs/traffic_signs/vietnamese_signs_{device}/")
    
    # Get best model path
    best_model = f"runs/traffic_signs/vietnamese_signs_{device}/weights/best.pt"
    if os.path.exists(best_model):
        print(f"🏆 Best model: {best_model}")
        
        # Copy to main directory for easy access
        import shutil
        dest_path = "models/vietnamese_traffic_signs.pt"
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_model, dest_path)
        print(f"✅ Model copied to: {dest_path}")
        
        # Update server config to use this model
        update_server_config(dest_path)
        
    return True

def update_server_config(model_path):
    """Update server configuration to use the trained model"""
    print("⚙️  Updating server configuration...")
    
    # Update app_demo.py to use trained model
    config_text = f'''
# Update this path in your server code:
TRAINED_MODEL_PATH = "{model_path}"

# To use in production, replace the mock detection with:
# model = YOLO(TRAINED_MODEL_PATH)
# results = model(image)
'''
    
    with open("TRAINED_MODEL_CONFIG.txt", "w") as f:
        f.write(config_text)
    
    print("✅ Configuration saved to TRAINED_MODEL_CONFIG.txt")

def main():
    """Main training function"""
    print("🚀 Vietnamese Traffic Signs - Quick Training")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    
    # Check dataset
    if not Path("data/dataset_config.yaml").exists():
        print("⚠️  Dataset not configured. Running setup first...")
        try:
            exec(open("setup_dataset.py").read())
        except:
            print("❌ Please run setup_dataset.py first!")
            return
    
    # Start training
    success = start_training()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Check training results in runs/traffic_signs/")
        print("2. Update server to use trained model")
        print("3. Test with web interface")
        print("4. Run: python server/app_demo.py")
        print("\n🌐 Access web interface at: http://localhost:8000")
    else:
        print("❌ Training failed. Check logs above.")

if __name__ == "__main__":
    main()