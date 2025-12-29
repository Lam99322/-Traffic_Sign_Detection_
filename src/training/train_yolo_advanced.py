#!/usr/bin/env python3
"""
Advanced YOLOv11 Training Script for Traffic Sign Detection
Features:
- GPU support with automatic device detection
- Advanced metrics (mAP, FPS, Confusion Matrix)
- Real-time training monitoring
- Automatic model optimization
- Export capabilities (ONNX, TensorRT)
- Learning rate scheduling
- Data augmentation
"""

import os
import sys
import time
import torch
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self, data_yaml_path, model_size='n', project_name='traffic_signs'):
        """
        Initialize YOLO trainer
        Args:
            data_yaml_path: Path to data.yaml file
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            project_name: Project name for tracking
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.model_size = model_size
        self.project_name = project_name
        
        # Check if data.yaml exists
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data yaml not found: {self.data_yaml_path}")
        
        # Load data configuration
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = None
        self.training_results = {}
        
        logger.info(f"🚀 YOLO Trainer initialized")
        logger.info(f"   Model: YOLOv11{model_size}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Classes: {self.data_config['nc']}")
        logger.info(f"   Dataset: {self.data_yaml_path}")
    
    def _setup_device(self):
        """Setup computing device with GPU support"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🔥 GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = 'cpu'
            logger.warning("⚠️  GPU not available, using CPU (will be slower)")
        
        return device
    
    def setup_directories(self):
        """Create training directories"""
        self.runs_dir = Path("runs/detect")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.runs_dir / f"train_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info(f"📁 Training directory: {self.run_dir}")
    
    def load_model(self):
        """Load YOLOv11 model"""
        model_name = f"yolo11{self.model_size}.pt"
        
        logger.info(f"📦 Loading {model_name}...")
        try:
            self.model = YOLO(model_name)
            logger.info(f"✅ Model loaded successfully")
            
            # Print model info
            self.model.info(verbose=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def train(self, epochs=100, imgsz=640, batch_size=16, lr0=0.01, 
              patience=50, save_period=10, **kwargs):
        """
        Train YOLO model with advanced configuration
        """
        logger.info("🚀 Starting training...")
        
        if self.model is None:
            self.load_model()
        
        # Training parameters
        train_args = {
            'data': str(self.data_yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'lr0': lr0,
            'device': self.device,
            'project': str(self.runs_dir),
            'name': self.run_dir.name,
            'save': True,
            'save_period': save_period,
            'patience': patience,
            'plots': True,
            'val': True,
            'exist_ok': True,
            
            # Advanced augmentations
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.3,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.9,
            'shear': 2.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            
            # Optimizer settings
            'optimizer': 'SGD',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            **kwargs
        }
        
        # Log training configuration
        logger.info("⚙️  Training Configuration:")
        for key, value in train_args.items():
            logger.info(f"   {key}: {value}")
        
        start_time = time.time()
        
        try:
            # Start training
            results = self.model.train(**train_args)
            
            training_time = time.time() - start_time
            
            # Save results
            self.training_results = {
                'training_time': training_time,
                'final_metrics': results.results_dict if hasattr(results, 'results_dict') else None,
                'best_model_path': str(results.save_dir / 'weights' / 'best.pt'),
                'last_model_path': str(results.save_dir / 'weights' / 'last.pt'),
                'config': train_args
            }
            
            logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
            logger.info(f"📁 Results saved to: {results.save_dir}")
            
            # Generate training report
            self.generate_training_report()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def evaluate(self, model_path=None):
        """
        Comprehensive model evaluation
        """
        logger.info("📊 Starting model evaluation...")
        
        if model_path is None:
            model_path = self.training_results.get('best_model_path')
        
        if model_path is None or not Path(model_path).exists():
            logger.error("❌ No model found for evaluation")
            return None
        
        # Load trained model
        model = YOLO(model_path)
        
        # Validation on test set
        logger.info("🔍 Evaluating on test set...")
        val_results = model.val(
            data=str(self.data_yaml_path),
            split='test',
            imgsz=640,
            plots=True,
            save_json=True,
            conf=0.001,  # Low confidence for complete evaluation
            iou=0.5
        )
        
        # Extract metrics
        metrics = {
            'mAP50': val_results.box.map50,
            'mAP50-95': val_results.box.map,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
            'f1_score': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr),
        }
        
        # Per-class metrics
        if hasattr(val_results.box, 'ap_class_index'):
            class_metrics = {}
            for i, class_idx in enumerate(val_results.box.ap_class_index):
                class_name = self.data_config['names'][int(class_idx)]
                class_metrics[class_name] = {
                    'ap50': val_results.box.ap50[i],
                    'ap': val_results.box.ap[i]
                }
            metrics['per_class'] = class_metrics
        
        logger.info("📈 Evaluation Results:")
        logger.info(f"   mAP@0.5: {metrics['mAP50']:.3f}")
        logger.info(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall: {metrics['recall']:.3f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.3f}")
        
        return metrics, val_results
    
    def benchmark_fps(self, model_path=None, test_images=100, imgsz=640):
        """
        Benchmark model FPS performance
        """
        logger.info("⚡ Benchmarking model FPS...")
        
        if model_path is None:
            model_path = self.training_results.get('best_model_path')
        
        model = YOLO(model_path)
        
        # Create dummy data
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model.predict(dummy_input, verbose=False)
        
        # Benchmark
        start_time = time.time()
        
        for _ in tqdm(range(test_images), desc="Benchmarking"):
            with torch.no_grad():
                _ = model.predict(dummy_input, verbose=False)
        
        total_time = time.time() - start_time
        fps = test_images / total_time
        
        logger.info(f"🚄 Performance Benchmark:")
        logger.info(f"   FPS: {fps:.2f}")
        logger.info(f"   Latency: {1000/fps:.2f} ms per frame")
        logger.info(f"   Device: {self.device}")
        
        return fps
    
    def export_model(self, model_path=None, formats=['onnx']):
        """
        Export model to different formats
        """
        logger.info(f"📤 Exporting model to {formats}...")
        
        if model_path is None:
            model_path = self.training_results.get('best_model_path')
        
        model = YOLO(model_path)
        exported_paths = {}
        
        for fmt in formats:
            try:
                exported_path = model.export(format=fmt, imgsz=640)
                exported_paths[fmt] = exported_path
                logger.info(f"✅ {fmt.upper()} export: {exported_path}")
            except Exception as e:
                logger.error(f"❌ {fmt.upper()} export failed: {e}")
        
        return exported_paths
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        report_path = self.run_dir / "training_report.md"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_content = f"""# YOLOv11 Training Report
        
## Training Summary
- **Timestamp**: {timestamp}
- **Model**: YOLOv11{self.model_size}
- **Device**: {self.device}
- **Dataset**: {self.data_config['nc']} classes
- **Training Time**: {self.training_results.get('training_time', 0)/3600:.2f} hours

## Dataset Information
- **Path**: {self.data_yaml_path}
- **Classes**: {list(self.data_config['names'].values())}
- **Splits**:
  - Train: {self.data_config['train']}
  - Val: {self.data_config['val']}
  - Test: {self.data_config['test']}

## Model Files
- **Best Model**: {self.training_results.get('best_model_path', 'N/A')}
- **Last Model**: {self.training_results.get('last_model_path', 'N/A')}

## Next Steps
1. **Evaluate Model**: Run evaluation to get detailed metrics
2. **Export Model**: Export to ONNX/TensorRT for deployment
3. **Deploy**: Integrate with web application

## Commands
```bash
# Evaluate model
python src/training/train_yolo.py --evaluate

# Export model
python src/training/train_yolo.py --export onnx

# Run web server
python server/app_demo.py
```
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 Training report saved: {report_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11 Training for Traffic Signs')
    parser.add_argument('--data', default='data/processed/data.yaml', 
                       help='Path to data.yaml file')
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=50, 
                       help='Early stopping patience')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run evaluation only')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Benchmark FPS')
    parser.add_argument('--export', nargs='+', default=[], 
                       choices=['onnx', 'engine', 'coreml', 'pb'],
                       help='Export formats')
    parser.add_argument('--model-path', 
                       help='Path to trained model for evaluation/export')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = YOLOTrainer(
            data_yaml_path=args.data,
            model_size=args.model,
            project_name='traffic_signs'
        )
        
        # Run evaluation only
        if args.evaluate:
            metrics, results = trainer.evaluate(args.model_path)
            return
        
        # Run benchmark only  
        if args.benchmark:
            fps = trainer.benchmark_fps(args.model_path)
            return
        
        # Export only
        if args.export:
            exported = trainer.export_model(args.model_path, args.export)
            return
        
        # Full training pipeline
        logger.info("🚀 Starting YOLO Training Pipeline...")
        
        # 1. Train model
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr,
            patience=args.patience
        )
        
        # 2. Evaluate model
        logger.info("📊 Evaluating trained model...")
        metrics, val_results = trainer.evaluate()
        
        # 3. Benchmark FPS
        logger.info("⚡ Benchmarking model performance...")
        fps = trainer.benchmark_fps()
        
        # 4. Export model (optional)
        if torch.cuda.is_available():
            logger.info("📤 Exporting model...")
            exported = trainer.export_model(formats=['onnx'])
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📈 Final mAP@0.5: {metrics['mAP50']:.3f}")
        logger.info(f"⚡ Performance: {fps:.1f} FPS")
        logger.info(f"📁 Results: {trainer.run_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()