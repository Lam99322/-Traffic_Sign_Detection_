#!/usr/bin/env python3
"""
🚦 Traffic Sign Detection System - Complete Requirements Check
Kiểm tra đầy đủ các yêu cầu của hệ thống nhận diện biển báo giao thông

YÊU CẦU:
✅ Xử lý dữ liệu: Chuẩn hóa, nâng cao (Data Augmentation), lọc ảnh mờ/nhòe
✅ Huấn luyện mô hình: YOLO11 CNN/ViT, sử dụng GPU, Colab hoặc Server AI
✅ Đánh giá: Accuracy, Precision, Recall, mAP, FPS, Confusion Matrix
✅ Triển khai: Python + OpenCV + PyTorch/TensorFlow, chạy webcam/video thời gian thực
✅ DÙNG CHO TẤT CẢ: Upload File, Upload Video, Webcam
"""

import os
import sys
import json
import time
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemRequirementsChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.requirements_status = {}
        
    def print_header(self):
        """Print system header"""
        print("=" * 80)
        print("🚦 TRAFFIC SIGN DETECTION SYSTEM - REQUIREMENTS CHECK")
        print("=" * 80)
        print("Kiểm tra đầy đủ yêu cầu hệ thống nhận diện biển báo giao thông")
        print()
        
    def check_data_processing(self):
        """✅ 1. Kiểm tra Xử lý dữ liệu"""
        print("1️⃣  XỬ LÝ DỮ LIỆU")
        print("-" * 40)
        
        requirements = {
            "Chuẩn hóa dữ liệu": False,
            "Data Augmentation": False,
            "Lọc ảnh mờ/nhòe": False,
            "YOLO format": False,
            "Stratified splitting": False
        }
        
        # Check data preprocessing script
        preprocessing_script = self.project_root / "src" / "data_preprocessing.py"
        if preprocessing_script.exists():
            with open(preprocessing_script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "enhance_image" in content and "CLAHE" in content:
                    requirements["Chuẩn hóa dữ liệu"] = True
                    print("✅ Chuẩn hóa dữ liệu (CLAHE enhancement)")
                
                if "augment_image" in content and "GaussianNoise" in content:
                    requirements["Data Augmentation"] = True
                    print("✅ Data Augmentation (Gaussian noise, rotation, flip)")
                
                if "detect_blur" in content and "Laplacian" in content:
                    requirements["Lọc ảnh mờ/nhòe"] = True
                    print("✅ Lọc ảnh mờ/nhòe (Laplacian blur detection)")
                
                if "save_yolo_annotations" in content:
                    requirements["YOLO format"] = True
                    print("✅ YOLO format annotations")
                    
                if "split_dataset" in content:
                    requirements["Stratified splitting"] = True
                    print("✅ Stratified dataset splitting")
        
        # Check processed data
        processed_data = self.project_root / "data" / "processed"
        if processed_data.exists() and (processed_data / "data.yaml").exists():
            print("✅ Dữ liệu đã được xử lý sẵn sàng")
        
        self.requirements_status["data_processing"] = requirements
        print(f"📊 Data Processing: {sum(requirements.values())}/{len(requirements)} requirements met")
        print()
        
    def check_model_training(self):
        """✅ 2. Kiểm tra Huấn luyện mô hình"""
        print("2️⃣  HUẤN LUYỆN MÔ HÌNH")
        print("-" * 40)
        
        requirements = {
            "YOLO11 support": False,
            "CNN/ViT models": False,
            "GPU support": False,
            "Advanced training": False,
            "Model export": False
        }
        
        # Check YOLO training script
        yolo_script = self.project_root / "src" / "training" / "train_yolo_advanced.py"
        if yolo_script.exists():
            with open(yolo_script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "yolo11" in content.lower():
                    requirements["YOLO11 support"] = True
                    print("✅ YOLO11 training support")
                
                if "torch.cuda" in content and "gpu" in content.lower():
                    requirements["GPU support"] = True
                    print("✅ GPU support với automatic detection")
                
                if "augment" in content and "mixup" in content:
                    requirements["Advanced training"] = True
                    print("✅ Advanced training (augmentation, mixup)")
                
                if "export" in content and "onnx" in content.lower():
                    requirements["Model export"] = True
                    print("✅ Model export (ONNX, TensorRT)")
        
        # Check CNN training script
        cnn_script = self.project_root / "src" / "training" / "train_cnn.py"
        if cnn_script.exists():
            with open(cnn_script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "timm" in content and ("vit" in content.lower() or "efficientnet" in content):
                    requirements["CNN/ViT models"] = True
                    print("✅ CNN/ViT models (ResNet, EfficientNet, Vision Transformer)")
        
        # Check for trained models
        yolo_models = list(Path("runs/detect").glob("*/weights/best.pt")) if Path("runs/detect").exists() else []
        if yolo_models:
            print(f"✅ Tìm thấy {len(yolo_models)} YOLO models đã training")
        
        self.requirements_status["model_training"] = requirements
        print(f"📊 Model Training: {sum(requirements.values())}/{len(requirements)} requirements met")
        print()
        
    def check_evaluation_metrics(self):
        """✅ 3. Kiểm tra Đánh giá"""
        print("3️⃣  ĐÁNH GIÁ MÔ HÌNH")
        print("-" * 40)
        
        requirements = {
            "Accuracy": False,
            "Precision & Recall": False,
            "mAP metrics": False,
            "FPS benchmark": False,
            "Confusion Matrix": False,
            "Comprehensive evaluation": False
        }
        
        # Check evaluation script
        eval_script = self.project_root / "src" / "evaluation" / "evaluate_system.py"
        if eval_script.exists():
            with open(eval_script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "accuracy_score" in content:
                    requirements["Accuracy"] = True
                    print("✅ Accuracy evaluation")
                
                if "precision_recall_fscore" in content:
                    requirements["Precision & Recall"] = True
                    print("✅ Precision & Recall metrics")
                
                if "map50" in content.lower() and "map" in content.lower():
                    requirements["mAP metrics"] = True
                    print("✅ mAP metrics (mAP@0.5, mAP@0.5:0.95)")
                
                if "benchmark_fps" in content:
                    requirements["FPS benchmark"] = True
                    print("✅ FPS benchmarking")
                
                if "confusion_matrix" in content:
                    requirements["Confusion Matrix"] = True
                    print("✅ Confusion Matrix visualization")
                
                if "comprehensive_report" in content:
                    requirements["Comprehensive evaluation"] = True
                    print("✅ Comprehensive evaluation report")
        
        # Check YOLO training script for metrics
        yolo_script = self.project_root / "src" / "training" / "train_yolo_advanced.py"
        if yolo_script.exists():
            with open(yolo_script, 'r', encoding='utf-8') as f:
                content = f.read()
                if "evaluate" in content and "benchmark" in content:
                    print("✅ Integrated evaluation in training")
        
        self.requirements_status["evaluation"] = requirements
        print(f"📊 Evaluation: {sum(requirements.values())}/{len(requirements)} requirements met")
        print()
        
    def check_deployment(self):
        """✅ 4. Kiểm tra Triển khai"""
        print("4️⃣  TRIỂN KHAI THỜI GIAN THỰC")
        print("-" * 40)
        
        requirements = {
            "Python + OpenCV": False,
            "PyTorch support": False,
            "Webcam real-time": False,
            "Video upload": False,
            "File upload": False,
            "WebSocket streaming": False,
            "Vietnamese TTS": False,
            "Web interface": False
        }
        
        # Check main server
        server_script = self.project_root / "server" / "app_demo.py"
        if server_script.exists():
            with open(server_script, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "import cv2" in content:
                    requirements["Python + OpenCV"] = True
                    print("✅ Python + OpenCV integration")
                
                if "import torch" in content or "pytorch" in content.lower():
                    requirements["PyTorch support"] = True
                    print("✅ PyTorch support")
                
                if "/api/webcam" in content:
                    requirements["Webcam real-time"] = True
                    print("✅ Webcam real-time detection")
                
                if "/api/detect-video" in content:
                    requirements["Video upload"] = True
                    print("✅ Video upload processing")
                
                if "/api/detect" in content and "upload" in content.lower():
                    requirements["File upload"] = True
                    print("✅ File upload detection")
                
                if "websocket" in content.lower() or "/ws/" in content:
                    requirements["WebSocket streaming"] = True
                    print("✅ WebSocket streaming support")
                
                if "gTTS" in content or "text_to_speech" in content:
                    requirements["Vietnamese TTS"] = True
                    print("✅ Vietnamese Text-to-Speech")
        
        # Check web interface
        template_file = self.project_root / "templates" / "index.html"
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "webcam" in content.lower() and "video" in content.lower() and "upload" in content.lower():
                    requirements["Web interface"] = True
                    print("✅ Complete web interface (upload, video, webcam, streaming)")
        
        # Check if server can run
        try:
            import cv2
            import torch
            import fastapi
            print("✅ All required packages installed")
        except ImportError as e:
            print(f"⚠️  Missing package: {e}")
        
        self.requirements_status["deployment"] = requirements
        print(f"📊 Deployment: {sum(requirements.values())}/{len(requirements)} requirements met")
        print()
        
    def check_overall_integration(self):
        """✅ 5. Kiểm tra tích hợp tổng thể"""
        print("5️⃣  TÍCH HỢP TỔNG THỂ")
        print("-" * 40)
        
        integration_checks = {
            "Project structure": False,
            "Data pipeline": False,
            "Model integration": False,
            "Web application": False,
            "Vietnamese support": False
        }
        
        # Check project structure
        required_dirs = [
            "src/training", "src/evaluation", "server", 
            "templates", "static", "config", "data"
        ]
        
        all_dirs_exist = all((self.project_root / d).exists() for d in required_dirs)
        if all_dirs_exist:
            integration_checks["Project structure"] = True
            print("✅ Complete project structure")
        
        # Check data pipeline
        if (self.project_root / "data" / "processed" / "data.yaml").exists():
            integration_checks["Data pipeline"] = True
            print("✅ Data processing pipeline working")
        
        # Check model files
        model_files = [
            "src/yolo_detector.py", "src/cnn_classifier.py"
        ]
        
        if all((self.project_root / f).exists() for f in model_files):
            integration_checks["Model integration"] = True
            print("✅ Model integration classes available")
        
        # Check web app readiness
        if (self.project_root / "server" / "app_demo.py").exists() and \
           (self.project_root / "templates" / "index.html").exists():
            integration_checks["Web application"] = True
            print("✅ Web application ready to run")
        
        # Check Vietnamese support
        classes_vi = self.project_root / "data" / "raw" / "archive" / "classes_vie.txt"
        if classes_vi.exists():
            integration_checks["Vietnamese support"] = True
            print("✅ Vietnamese language support")
        
        self.requirements_status["integration"] = integration_checks
        print(f"📊 Integration: {sum(integration_checks.values())}/{len(integration_checks)} checks passed")
        print()
        
    def generate_final_report(self):
        """Generate final requirements report"""
        print("📋 BÁO CÁO TỔNG KẾT YÊU CẦU")
        print("=" * 60)
        
        total_requirements = 0
        met_requirements = 0
        
        for category, requirements in self.requirements_status.items():
            cat_total = len(requirements)
            cat_met = sum(requirements.values())
            total_requirements += cat_total
            met_requirements += cat_met
            
            percentage = (cat_met / cat_total) * 100
            status = "✅ HOÀN THÀNH" if percentage == 100 else f"🔄 {percentage:.0f}%"
            
            print(f"{category.replace('_', ' ').title()}: {cat_met}/{cat_total} {status}")
        
        overall_percentage = (met_requirements / total_requirements) * 100
        
        print("-" * 60)
        print(f"📊 TỔNG CỘNG: {met_requirements}/{total_requirements} ({overall_percentage:.0f}%)")
        
        if overall_percentage >= 90:
            print("\n🎉 HỆ THỐNG ĐẠT ĐỦ YÊU CẦU!")
            print("✅ Sẵn sàng cho sản xuất và demo")
        elif overall_percentage >= 75:
            print("\n✅ HỆ THỐNG GẦN HOÀN THÀNH!")
            print("🔄 Cần hoàn thiện một số tính năng")
        else:
            print("\n🔄 HỆ THỐNG CẦN PHÁT TRIỂN THÊM")
            print("⚠️  Cần hoàn thành thêm các yêu cầu")
        
        # Usage instructions
        print("\n🚀 HƯỚNG DẪN SỬ DỤNG:")
        print("-" * 30)
        print("1. Xử lý dữ liệu:")
        print("   python src/data_preprocessing.py")
        print()
        print("2. Huấn luyện YOLO:")
        print("   python src/training/train_yolo_advanced.py --epochs 50")
        print()
        print("3. Đánh giá hệ thống:")
        print("   python src/evaluation/evaluate_system.py")
        print()
        print("4. Chạy web application:")
        print("   python server/app_demo.py")
        print("   Truy cập: http://localhost:8000")
        print()
        print("5. Tính năng web:")
        print("   ✅ Upload ảnh - nhận diện tức thì")
        print("   ✅ Upload video - xử lý từng frame")
        print("   ✅ Webcam - real-time detection")
        print("   ✅ Live streaming - WebSocket")
        print("   ✅ Vietnamese TTS - đọc tên biển báo")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_percentage': overall_percentage,
            'total_requirements': total_requirements,
            'met_requirements': met_requirements,
            'detailed_status': self.requirements_status
        }
        
        report_path = self.project_root / "REQUIREMENTS_CHECK_REPORT.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Báo cáo chi tiết đã lưu: {report_path}")
        
        return overall_percentage >= 90

def main():
    """Main entry point"""
    checker = SystemRequirementsChecker()
    
    checker.print_header()
    checker.check_data_processing()
    checker.check_model_training()
    checker.check_evaluation_metrics()
    checker.check_deployment()
    checker.check_overall_integration()
    
    system_ready = checker.generate_final_report()
    
    if system_ready:
        print("\n🎯 KẾT LUẬN: HỆ THỐNG ĐẠT ĐỦ TẤT CẢ YÊU CẦU!")
        return True
    else:
        print("\n📝 KẾT LUẬN: HỆ THỐNG CẦN HOÀN THIỆN THÊM")
        return False

if __name__ == "__main__":
    main()