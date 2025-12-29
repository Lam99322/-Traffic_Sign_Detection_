
# 🏗️ Traffic Sign Detection - Clean Project Structure

## 📁 Organized Directory Structure

```
Traffic_Sign_Detection-main/
├── 🖥️ Core Application
│   ├── server/
│   │   ├── app_demo.py          # Main FastAPI server
│   │   └── __init__.py
│   ├── templates/
│   │   └── index.html           # Web interface
│   └── static/
│       ├── css/
│       └── js/

├── 🧠 AI/ML Components  
│   ├── src/
│   │   ├── yolo_detector.py     # YOLO detection
│   │   ├── cnn_classifier.py    # CNN classification
│   │   ├── data_preprocessing.py # Data processing
│   │   ├── training/
│   │   │   ├── train_cnn.py
│   │   │   └── train_yolo.py
│   │   ├── evaluation/
│   │   │   └── evaluate_system.py
│   │   └── utils/
│   │       ├── metrics.py
│   │       └── visualization.py

├── 📊 Dataset & Training
│   ├── data/
│   │   ├── dataset_config.yaml   # Dataset configuration
│   │   ├── class_mapping.json    # Vietnamese class names
│   │   ├── train/               # Training data
│   │   ├── valid/               # Validation data
│   │   └── test/                # Test data
│   ├── quick_train.py           # Quick training script
│   ├── train_vietnamese_signs.py # Detailed training
│   └── setup_dataset.py         # Dataset setup

├── ⚙️ Configuration
│   ├── config/
│   │   ├── logging_config.yaml
│   │   ├── server_config.yaml
│   │   └── yolo_config.yaml
│   ├── requirements.txt
│   └── setup.py

└── 🧪 Testing & Utils
    ├── tests/
    │   ├── test_classifier.py
    │   └── test_detector.py
    ├── test_web.py             # Web interface test
    ├── fix_js.py              # JavaScript fixes
    └── setup_project.py       # Project setup
```

## 🎯 Core Files (Essential)

### 1. **Server & Web Interface**
- `server/app_demo.py` - Main FastAPI application
- `templates/index.html` - Web interface
- `static/` - CSS/JS assets

### 2. **AI/ML Core**
- `src/yolo_detector.py` - YOLO detection engine
- `src/cnn_classifier.py` - CNN classification
- `src/data_preprocessing.py` - Data processing pipeline

### 3. **Training Pipeline**
- `quick_train.py` - Quick model training
- `train_vietnamese_signs.py` - Full training pipeline
- `setup_dataset.py` - Dataset configuration

### 4. **Dataset**
- `data/` - Vietnamese traffic signs dataset (4,803 images)
- `data/dataset_config.yaml` - Training configuration
- `data/class_mapping.json` - Vietnamese class names

## 🚀 Usage Workflow

1. **Setup**: `python setup_project.py`
2. **Dataset**: `python setup_dataset.py`
3. **Train**: `python quick_train.py`
4. **Run**: `python server/app_demo.py`
5. **Test**: http://localhost:8000

## 🧹 Cleaned Up (Removed)

- ❌ Duplicate training scripts
- ❌ Old model files (.pt)
- ❌ Unused server files
- ❌ Cache directories
- ❌ Demo/test duplicates
