#!/usr/bin/env python3
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
    print(f"🖥️  Using device: {device}")
    
    # Load dataset config
    dataset_config = "data\dataset_config.yaml"
    
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
    
    print("\n📊 Training Results:")
    print(f"📈 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"📈 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    # Export model
    print("\n📦 Exporting model...")
    model.export(format='onnx')
    print("✅ Model exported to ONNX format")
    
    return model, results

if __name__ == "__main__":
    model, results = train_vietnamese_traffic_signs()
