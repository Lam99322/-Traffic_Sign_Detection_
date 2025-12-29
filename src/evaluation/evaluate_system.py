#!/usr/bin/env python3
"""
Comprehensive Evaluation Dashboard for Traffic Sign Detection System
Features:
- YOLO and CNN model evaluation
- Performance metrics (mAP, FPS, Accuracy, Precision, Recall)
- Confusion matrix visualization
- Real-time performance benchmarks
- Model comparison and analysis
- Export evaluation reports
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficSignEvaluator:
    def __init__(self, data_dir="data/processed", results_dir="evaluation_results"):
        """
        Initialize evaluator
        Args:
            data_dir: Path to processed dataset
            results_dir: Directory to save evaluation results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load class mapping
        classes_file = self.data_dir.parent / "raw" / "archive" / "classes.txt"
        self.classes = {}
        if classes_file.exists():
            with open(classes_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    self.classes[i] = line.strip()
        else:
            self.classes = {i: f"class_{i}" for i in range(15)}
        
        # Vietnamese class names
        self.class_names_vi = {
            0: "Giới hạn tốc độ 20",
            1: "Giới hạn tốc độ 30", 
            2: "Giới hạn tốc độ 50",
            3: "Giới hạn tốc độ 60",
            4: "Giới hạn tốc độ 70",
            5: "Giới hạn tốc độ 80",
            6: "Cấm vượt",
            7: "Cấm đi vào",
            8: "Nguy hiểm",
            9: "Bắt buộc rẽ trái",
            10: "Bắt buộc rẽ phải", 
            11: "Bắt buộc đi thẳng",
            12: "Dừng lại",
            13: "Nhường đường",
            14: "Đường ưu tiên"
        }
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔥 Using device: {self.device}")
        
        # Results storage
        self.evaluation_results = {}
        
    def find_latest_models(self):
        """Find latest trained models"""
        models_found = {}
        
        # Find YOLO models
        yolo_runs = Path("runs/detect")
        if yolo_runs.exists():
            for run_dir in sorted(yolo_runs.glob("train_*"), reverse=True):
                best_model = run_dir / "weights" / "best.pt"
                if best_model.exists():
                    models_found['yolo'] = best_model
                    break
        
        # Find CNN models
        cnn_runs = Path("runs/classify")
        if cnn_runs.exists():
            for run_dir in sorted(cnn_runs.glob("train_*"), reverse=True):
                best_model = run_dir / "best_model.pth"
                if best_model.exists():
                    models_found['cnn'] = best_model
                    break
        
        return models_found
    
    def evaluate_yolo_model(self, model_path):
        """Evaluate YOLO model performance"""
        logger.info(f"📊 Evaluating YOLO model: {model_path}")
        
        try:
            # Load model
            model = YOLO(str(model_path))
            
            # Run validation
            results = model.val(
                data=str(self.data_dir / "data.yaml"),
                split='test',
                imgsz=640,
                conf=0.001,
                iou=0.5,
                plots=True,
                save_json=True
            )
            
            # Extract metrics
            metrics = {
                'model_type': 'YOLO',
                'model_path': str(model_path),
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': float(2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-8)),
                'inference_speed': results.speed['inference'],
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
            }
            
            # Per-class AP
            if hasattr(results.box, 'ap_class_index') and len(results.box.ap_class_index) > 0:
                class_ap = {}
                for i, class_idx in enumerate(results.box.ap_class_index):
                    class_name = self.classes.get(int(class_idx), f'class_{class_idx}')
                    class_ap[class_name] = {
                        'ap50': float(results.box.ap50[i]),
                        'ap50_95': float(results.box.ap[i])
                    }
                metrics['per_class_ap'] = class_ap
            
            logger.info(f"✅ YOLO Evaluation Results:")
            logger.info(f"   mAP@0.5: {metrics['mAP50']:.3f}")
            logger.info(f"   mAP@0.5:0.95: {metrics['mAP50_95']:.3f}")
            logger.info(f"   Precision: {metrics['precision']:.3f}")
            logger.info(f"   Recall: {metrics['recall']:.3f}")
            logger.info(f"   F1-Score: {metrics['f1_score']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ YOLO evaluation failed: {e}")
            return None
    
    def benchmark_fps(self, model_path, model_type='yolo', test_images=100):
        """Benchmark model FPS"""
        logger.info(f"⚡ Benchmarking {model_type} FPS...")
        
        try:
            if model_type == 'yolo':
                model = YOLO(str(model_path))
                
                # Create dummy data
                dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Warmup
                for _ in range(10):
                    _ = model.predict(dummy_input, verbose=False)
                
                # Benchmark
                start_time = time.time()
                for _ in range(test_images):
                    _ = model.predict(dummy_input, verbose=False)
                total_time = time.time() - start_time
                
                fps = test_images / total_time
                latency = 1000 / fps  # ms
                
            elif model_type == 'cnn':
                # For CNN models, we'd need to implement similar benchmarking
                fps = 0  # Placeholder
                latency = 0
            
            logger.info(f"🚄 {model_type.upper()} Performance:")
            logger.info(f"   FPS: {fps:.2f}")
            logger.info(f"   Latency: {latency:.2f} ms")
            
            return {'fps': fps, 'latency': latency}
            
        except Exception as e:
            logger.error(f"❌ FPS benchmark failed: {e}")
            return {'fps': 0, 'latency': 0}
    
    def analyze_test_predictions(self, model_path, model_type='yolo'):
        """Analyze predictions on test set"""
        logger.info(f"🔍 Analyzing {model_type} predictions...")
        
        test_images_dir = self.data_dir / "test" / "images"
        test_labels_dir = self.data_dir / "test" / "labels"
        
        if not test_images_dir.exists():
            logger.error("❌ Test images directory not found")
            return None
        
        predictions = []
        ground_truths = []
        
        try:
            if model_type == 'yolo':
                model = YOLO(str(model_path))
                
                for img_path in tqdm(list(test_images_dir.glob("*.jpg")), desc="Analyzing"):
                    # Get ground truth
                    label_path = test_labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                gt_class = int(line.split()[0])
                                ground_truths.append(gt_class)
                                
                                # Get prediction
                                results = model.predict(str(img_path), verbose=False)
                                if len(results) > 0 and len(results[0].boxes) > 0:
                                    # Get highest confidence prediction
                                    boxes = results[0].boxes
                                    best_idx = torch.argmax(boxes.conf)
                                    pred_class = int(boxes.cls[best_idx])
                                    predictions.append(pred_class)
                                else:
                                    predictions.append(-1)  # No detection
        
        except Exception as e:
            logger.error(f"❌ Prediction analysis failed: {e}")
            return None
        
        # Calculate metrics
        if len(predictions) > 0 and len(ground_truths) > 0:
            # Filter out no-detections for classification metrics
            valid_indices = [i for i, p in enumerate(predictions) if p != -1]
            valid_preds = [predictions[i] for i in valid_indices]
            valid_gt = [ground_truths[i] for i in valid_indices]
            
            if len(valid_preds) > 0:
                accuracy = accuracy_score(valid_gt, valid_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    valid_gt, valid_preds, average='weighted', zero_division=0
                )
                cm = confusion_matrix(valid_gt, valid_preds)
                
                analysis = {
                    'total_samples': len(predictions),
                    'detected_samples': len(valid_preds),
                    'detection_rate': len(valid_preds) / len(predictions),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist()
                }
                
                logger.info(f"📈 Classification Analysis:")
                logger.info(f"   Detection Rate: {analysis['detection_rate']:.3f}")
                logger.info(f"   Accuracy: {analysis['accuracy']:.3f}")
                logger.info(f"   Precision: {analysis['precision']:.3f}")
                logger.info(f"   Recall: {analysis['recall']:.3f}")
                logger.info(f"   F1-Score: {analysis['f1_score']:.3f}")
                
                return analysis
        
        return None
    
    def create_comprehensive_report(self):
        """Create comprehensive evaluation report"""
        logger.info("📋 Creating comprehensive evaluation report...")
        
        # Find models
        models = self.find_latest_models()
        
        if not models:
            logger.error("❌ No trained models found!")
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'classes': self.classes,
                'num_classes': len(self.classes),
                'data_dir': str(self.data_dir)
            },
            'models_evaluated': {}
        }
        
        # Evaluate each model
        for model_type, model_path in models.items():
            logger.info(f"🔬 Evaluating {model_type} model...")
            
            model_results = {
                'model_path': str(model_path),
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
            }
            
            if model_type == 'yolo':
                # YOLO evaluation
                yolo_metrics = self.evaluate_yolo_model(model_path)
                if yolo_metrics:
                    model_results.update(yolo_metrics)
                
                # FPS benchmark
                fps_results = self.benchmark_fps(model_path, 'yolo')
                model_results.update(fps_results)
                
                # Prediction analysis
                analysis = self.analyze_test_predictions(model_path, 'yolo')
                if analysis:
                    model_results['classification_analysis'] = analysis
            
            report['models_evaluated'][model_type] = model_results
        
        # Save report
        report_path = self.results_dir / "comprehensive_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Create visualizations
        self.create_visualizations(report)
        
        # Generate markdown report
        self.generate_markdown_report(report)
        
        logger.info(f"✅ Comprehensive report saved to: {self.results_dir}")
        
        return report
    
    def create_visualizations(self, report):
        """Create evaluation visualizations"""
        logger.info("📊 Creating visualizations...")
        
        # Model comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(report['models_evaluated'].keys())
        if len(models) == 0:
            return
        
        # 1. Performance metrics comparison
        metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
        model_metrics = {}
        
        for model in models:
            model_data = report['models_evaluated'][model]
            model_metrics[model] = [model_data.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model, values) in enumerate(model_metrics.items()):
            axes[0, 0].bar(x + i * width, values, width, label=model, alpha=0.8)
        
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x + width / 2)
        axes[0, 0].set_xticklabels(metrics, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. FPS and Model Size
        if len(models) > 0:
            fps_values = [report['models_evaluated'][m].get('fps', 0) for m in models]
            size_values = [report['models_evaluated'][m].get('model_size_mb', 0) for m in models]
            
            ax2 = axes[0, 1]
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar([m + '\nFPS' for m in models], fps_values, alpha=0.7, color='skyblue')
            bars2 = ax2_twin.bar([m + '\nSize' for m in models], size_values, alpha=0.7, color='lightcoral')
            
            ax2.set_ylabel('FPS', color='blue')
            ax2_twin.set_ylabel('Model Size (MB)', color='red')
            ax2.set_title('Performance vs Model Size')
        
        # 3. Confusion Matrix (if available)
        for i, model in enumerate(models):
            analysis = report['models_evaluated'][model].get('classification_analysis')
            if analysis and 'confusion_matrix' in analysis:
                cm = np.array(analysis['confusion_matrix'])
                
                ax_idx = (1, 0) if i == 0 else (1, 1)
                if ax_idx[1] < 2:  # Only plot if we have space
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[ax_idx])
                    axes[ax_idx].set_title(f'{model.upper()} Confusion Matrix')
                    axes[ax_idx].set_xlabel('Predicted')
                    axes[ax_idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'evaluation_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Visualizations saved")
    
    def generate_markdown_report(self, report):
        """Generate markdown evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# Traffic Sign Detection System - Evaluation Report

**Generated:** {timestamp}
**Dataset:** {report['dataset_info']['num_classes']} classes
**Data Directory:** {report['dataset_info']['data_dir']}

## 📊 Summary

"""
        
        # Add model results
        for model_type, results in report['models_evaluated'].items():
            markdown_content += f"""
### {model_type.upper()} Model Results

- **Model Path:** `{results['model_path']}`
- **Model Size:** {results['model_size_mb']:.2f} MB
"""
            
            if 'mAP50' in results:
                markdown_content += f"""
- **mAP@0.5:** {results['mAP50']:.3f}
- **mAP@0.5:0.95:** {results['mAP50_95']:.3f}
- **Precision:** {results['precision']:.3f}
- **Recall:** {results['recall']:.3f}
- **F1-Score:** {results['f1_score']:.3f}
"""
            
            if 'fps' in results:
                markdown_content += f"""
- **FPS:** {results['fps']:.2f}
- **Latency:** {results['latency']:.2f} ms
"""
            
            if 'classification_analysis' in results:
                analysis = results['classification_analysis']
                markdown_content += f"""
- **Detection Rate:** {analysis['detection_rate']:.3f}
- **Classification Accuracy:** {analysis['accuracy']:.3f}
"""
        
        markdown_content += f"""
## 📈 Class Information

{chr(10).join([f"- **{i}:** {name} ({self.class_names_vi.get(i, name)})" for i, name in self.classes.items()])}

## 🚀 Usage

### Run Web Application
```bash
python server/app_demo.py
```

### Train New Models
```bash
# YOLO training
python src/training/train_yolo_advanced.py --epochs 50

# CNN training (when available)
python src/training/train_cnn.py --epochs 50
```

### Evaluate Models
```bash
python src/evaluation/evaluate_system.py
```

## 📁 Files Generated

- `evaluation_charts.png` - Performance comparison charts
- `comprehensive_evaluation_report.json` - Detailed JSON report
- This markdown report

---
*Generated by Traffic Sign Detection System*
"""
        
        report_path = self.results_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"📋 Markdown report saved: {report_path}")

def main():
    """Main evaluation entry point"""
    logger.info("🚀 Starting Traffic Sign Detection System Evaluation")
    logger.info("=" * 60)
    
    try:
        evaluator = TrafficSignEvaluator()
        report = evaluator.create_comprehensive_report()
        
        if report:
            logger.info("=" * 60)
            logger.info("✅ EVALUATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("📋 Results available in: evaluation_results/")
            logger.info("📊 Charts: evaluation_charts.png")
            logger.info("📄 Report: evaluation_report.md")
            logger.info("📋 JSON: comprehensive_evaluation_report.json")
        else:
            logger.error("❌ Evaluation failed")
            
    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()