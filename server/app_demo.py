from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import yaml
import cv2
import numpy as np
import os
from pathlib import Path
import logging
import base64
import tempfile
import time
import json
import asyncio
from typing import List
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hybrid_detector import HybridTrafficSignDetector

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Vietnamese traffic signs class mapping
def load_class_mapping():
    """Load Vietnamese class names from dataset"""
    try:
        mapping_path = Path("data/class_mapping.json")
        if mapping_path.exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                # Convert string keys to int
                return {int(k): v for k, v in mapping.items()}
    except Exception as e:
        logger.warning(f"Could not load class mapping: {e}")
    
    # Default Vietnamese traffic signs mapping
    return {
        0: {"vi": "Cấm con lái", "en": "No passing", "description": "Biển báo cấm các xe cơ giới vượt"},
        1: {"vi": "Cấm dừng và đỗ", "en": "No stopping and parking", "description": "Biển báo cấm dừng xe và đỗ xe"},
        2: {"vi": "Cấm ngược chiều", "en": "No entry", "description": "Biển báo cấm đi ngược chiều"},
        3: {"vi": "Cấm rẽ", "en": "No turn", "description": "Biển báo cấm rẽ trái hoặc rẽ phải"},
        4: {"vi": "Giới hạn tốc độ", "en": "Speed limit", "description": "Biển báo giới hạn tốc độ tối đa"},
        5: {"vi": "Hiệu lệnh", "en": "Mandatory", "description": "Biển báo hiệu lệnh bắt buộc"},
        6: {"vi": "Nguy hiểm", "en": "Warning", "description": "Biển báo cảnh báo nguy hiểm"}
    }

# Global class mapping
CLASS_MAPPING = load_class_mapping()

# Load config
config_path = "config/server_config.yaml"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    # Default config if file doesn't exist
    config = {
        'server': {'host': '0.0.0.0', 'port': 8000, 'debug': False, 'reload': False},
        'processing': {
            'max_image_size': 5242880, 
            'max_video_size': 52428800,  # 50MB
            'allowed_extensions': ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
        },
        'cors': {'allow_origins': ['*'], 'allow_methods': ['*'], 'allow_headers': ['*']}
    }

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Sign Detection API",
    description="API for detecting and classifying traffic signs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['cors']['allow_origins'],
    allow_credentials=True,
    allow_methods=config['cors']['allow_methods'],
    allow_headers=config['cors']['allow_headers'],
)

# Initialize hybrid detector
try:
    hybrid_detector = HybridTrafficSignDetector(
        confidence_threshold=0.5,
        use_vit=False  # Start with CNN, can switch to ViT later
    )
    logger.info("Hybrid detector initialized successfully")
    MODELS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not initialize hybrid detector: {e}. Running in demo mode.")
    hybrid_detector = None
    MODELS_AVAILABLE = False

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")
else:
    templates = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>Traffic Sign Detection</title></head>
            <body>
                <h1>🚦 Traffic Sign Detection System</h1>
                <p>Upload an image to detect traffic signs!</p>
                <form action="/api/detect" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <button type="submit">Detect Signs</button>
                </form>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Traffic Sign Detection API is running",
        "models_available": MODELS_AVAILABLE
    }

@app.post("/api/detect")
async def detect_traffic_signs(file: UploadFile = File(...)):
    """
    Traffic sign detection using hybrid YOLOv11 + CNN/ViT pipeline
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        if len(contents) > config['processing']['max_image_size']:
            raise HTTPException(status_code=400, detail="Image size too large")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        height, width = image.shape[:2]
        
        # Use hybrid detector if available
        if MODELS_AVAILABLE and hybrid_detector is not None:
            try:
                # Real detection with hybrid pipeline
                results = hybrid_detector.detect_and_classify(image)
                
                if results['errors']:
                    logger.warning(f"Detection errors: {results['errors']}")
                
                # Format results for API response
                formatted_detections = []
                for classification in results.get('classifications', []):
                    bbox = classification['bbox']
                    class_id = classification['class_id']
                    class_info = CLASS_MAPPING.get(class_id, CLASS_MAPPING[0])
                    
                    formatted_detections.append({
                        'bbox': bbox,
                        'detection_confidence': classification['detection_confidence'],
                        'classification_confidence': classification['classification_confidence'],
                        'combined_confidence': classification['combined_confidence'],
                        'class_id': class_id,
                        'class_name': class_info['en'],
                        'class_name_vi': class_info['vi'],
                        'description': class_info['description'],
                        'model_type': classification.get('model_type', 'CNN'),
                        'note': 'Real hybrid detection: YOLOv11 + CNN/ViT'
                    })
                
                return JSONResponse(content={
                    "success": True,
                    "message": f"Detected {results['total_signs']} traffic signs using hybrid pipeline",
                    "num_detections": len(formatted_detections),
                    "detections": formatted_detections,
                    "processing_time": results['processing_time'],
                    "model_info": hybrid_detector.get_model_info(),
                    "image_size": {"height": height, "width": width}
                })
                
            except Exception as e:
                logger.error(f"Hybrid detection failed: {e}")
                # Fall back to demo mode
                pass
        
        # Demo mode - Mock detection results
        class_id = np.random.randint(0, 7)  # Random class from our 7 classes
        class_info = CLASS_MAPPING.get(class_id, CLASS_MAPPING[0])
        
        mock_detections = [{
            'bbox': [int(width*0.1), int(height*0.1), int(width*0.3), int(height*0.3)],
            'detection_confidence': round(np.random.uniform(0.75, 0.95), 2),
            'classification_confidence': round(np.random.uniform(0.75, 0.95), 2),
            'combined_confidence': round(np.random.uniform(0.75, 0.95), 2),
            'class_id': class_id,
            'class_name': class_info['en'],
            'class_name_vi': class_info['vi'],
            'description': class_info['description'],
            'model_type': 'Demo',
            'note': 'Demo mode - Train models for actual detection'
        }]
        
        return JSONResponse(content={
            "success": True,
            "message": "Demo mode - train YOLO and CNN models for actual detection",
            "num_detections": len(mock_detections),
            "detections": mock_detections,
            "image_size": {"height": height, "width": width}
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/classes")
async def get_classes():
    """Get list of supported traffic sign classes with Vietnamese names from dataset"""
    # Use the actual class mapping from our Vietnamese dataset
    classes = CLASS_MAPPING
    
    return JSONResponse(content={
        "success": True,
        "num_classes": len(classes),
        "classes": classes
    })

@app.post("/api/webcam-capture")
async def webcam_capture(image_data: str = Form(...)):
    """
    Process webcam captured image
    """
    try:
        # Remove data:image/jpeg;base64, prefix
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        height, width = image.shape[:2]
        
        # Mock detection với Vietnamese dataset classes
        class_id = np.random.randint(0, 7)
        class_info = CLASS_MAPPING.get(class_id, CLASS_MAPPING[0])
        
        mock_detections = [{
            'bbox': [int(width*0.2), int(height*0.2), int(width*0.6), int(height*0.6)],
            'confidence': round(np.random.uniform(0.80, 0.95), 2),
            'class_id': class_id,
            'class_name_en': class_info['en'],
            'class_name_vi': class_info['vi'],
            'description': class_info['description'],
            'note': 'Webcam capture - Vietnamese dataset demo'
        }]
        
        return JSONResponse(content={
            "success": True,
            "source": "webcam",
            "message": "Nhận diện thành công từ webcam",
            "num_detections": len(mock_detections),
            "detections": mock_detections,
            "image_size": {"height": height, "width": width}
        })
        
    except Exception as e:
        logger.error(f"Error processing webcam image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """
    Process video file for traffic sign detection
    """
    try:
        # Validate video file
        if not any(file.filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
            raise HTTPException(status_code=400, detail="File must be a video (.mp4, .avi, .mov, .mkv)")
        
        # Check file size
        contents = await file.read()
        if len(contents) > config['processing']['max_video_size']:
            raise HTTPException(status_code=400, detail="Video file too large (max 50MB)")
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(contents)
            video_path = tmp_file.name
        
        try:
            # Process video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frames every second for detection
            frame_detections = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every 30th frame (approximately 1 per second for 30fps video)
                if frame_count % 30 == 0:
                    height, width = frame.shape[:2]
                    timestamp = frame_count / fps
                    
                    # Mock detection for demo
                    mock_detection = {
                        'timestamp': round(timestamp, 2),
                        'frame_number': frame_count,
                        'detections': [{
                            'bbox': [int(width*0.1), int(height*0.1), int(width*0.4), int(height*0.4)],
                            'confidence': 0.88,
                            'class_id': 12,
                            'class_name_en': 'stop',
                            'class_name_vi': 'Biển báo dừng lại'
                        }]
                    }
                    frame_detections.append(mock_detection)
                
                frame_count += 1
            
            cap.release()
            
            return JSONResponse(content={
                "success": True,
                "source": "video",
                "message": f"Xử lý video thành công - {len(frame_detections)} khung hình được phân tích",
                "video_info": {
                    "filename": file.filename,
                    "total_frames": total_frames,
                    "duration_seconds": round(duration, 2),
                    "fps": fps
                },
                "frame_detections": frame_detections,
                "summary": {
                    "total_detections": sum(len(fd['detections']) for fd in frame_detections),
                    "detected_classes": list(set(d['class_name_vi'] for fd in frame_detections for d in fd['detections']))
                }
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/setup-guide")
async def setup_guide():
    """Get setup instructions"""
    return JSONResponse(content={
        "success": True,
        "setup_steps": [
            "1. Add dataset to data/raw/archive/ folder",
            "2. Run: python src/data_preprocessing.py",
            "3. Train YOLO: python src/training/train_yolo.py",
            "4. Train CNN: python src/training/train_cnn.py",
            "5. Update model paths in config/server_config.yaml",
            "6. Restart server for full functionality"
        ],
        "dataset_structure": {
            "data/raw/archive/": {
                "images/": "Image files (.jpg, .jpeg, .png)",
                "labels/": "YOLO format annotation files (.txt)",
                "classes.txt": "Class names file"
            }
        }
    })

@app.websocket("/ws/realtime-detection")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            if frame_data.get("type") == "frame":
                # Process frame
                image_data = frame_data.get("image_data", "")
                
                try:
                    # Remove data:image/jpeg;base64, prefix
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        height, width = image.shape[:2]
                        
                        # Mock real-time detection
                        detections = []
                        if np.random.random() > 0.3:  # 70% chance of detection
                            class_id = np.random.randint(0, 15)
                            classes_map = {
                                0: {"en": "speed_limit_20", "vi": "Giới hạn tốc độ 20 km/h"},
                                1: {"en": "speed_limit_30", "vi": "Giới hạn tốc độ 30 km/h"},
                                2: {"en": "speed_limit_50", "vi": "Giới hạn tốc độ 50 km/h"},
                                3: {"en": "speed_limit_60", "vi": "Giới hạn tốc độ 60 km/h"},
                                4: {"en": "speed_limit_70", "vi": "Giới hạn tốc độ 70 km/h"},
                                5: {"en": "speed_limit_80", "vi": "Giới hạn tốc độ 80 km/h"},
                                6: {"en": "no_overtaking", "vi": "Cấm vượt"},
                                7: {"en": "no_entry", "vi": "Cấm đi vào"},
                                8: {"en": "danger", "vi": "Nguy hiểm"},
                                9: {"en": "mandatory_left", "vi": "Bắt buộc rẽ trái"},
                                10: {"en": "mandatory_right", "vi": "Bắt buộc rẽ phải"},
                                11: {"en": "mandatory_straight", "vi": "Bắt buộc đi thẳng"},
                                12: {"en": "stop", "vi": "Dừng lại"},
                                13: {"en": "yield", "vi": "Nhường đường"},
                                14: {"en": "priority_road", "vi": "Đường ưu tiên"}
                            }
                            
                            detection = {
                                'bbox': [
                                    int(width * np.random.uniform(0.1, 0.3)),
                                    int(height * np.random.uniform(0.1, 0.3)),
                                    int(width * np.random.uniform(0.4, 0.7)),
                                    int(height * np.random.uniform(0.4, 0.7))
                                ],
                                'confidence': round(np.random.uniform(0.7, 0.95), 2),
                                'class_id': class_id,
                                'class_name_en': classes_map[class_id]["en"],
                                'class_name_vi': classes_map[class_id]["vi"],
                                'timestamp': time.time()
                            }
                            detections.append(detection)
                        
                        # Send detection results
                        response = {
                            "type": "detection_result",
                            "timestamp": time.time(),
                            "detections": detections,
                            "image_size": {"width": width, "height": height},
                            "fps": frame_data.get("fps", 0)
                        }
                        
                        await manager.send_personal_message(json.dumps(response), websocket)
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    error_response = {
                        "type": "error",
                        "message": f"Error processing frame: {str(e)}"
                    }
                    await manager.send_personal_message(json.dumps(error_response), websocket)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed")

@app.post("/api/start-video-stream")
async def start_video_stream():
    """
    Start video streaming session
    """
    return JSONResponse(content={
        "success": True,
        "message": "Video streaming session started",
        "websocket_url": "/ws/realtime-detection",
        "instructions": [
            "1. Connect to WebSocket endpoint",
            "2. Send frames as base64 encoded images",
            "3. Receive real-time detection results",
            "4. Format: {'type': 'frame', 'image_data': 'base64_string', 'fps': 30}"
        ]
    })

if __name__ == "__main__":
    server_config = config['server']
    print(f"🚦 Starting Traffic Sign Detection Server...")
    print(f"📍 Server will run at: http://{server_config['host']}:{server_config['port']}")
    print(f"📚 API Documentation: http://{server_config['host']}:{server_config['port']}/docs")
    print(f"📊 Health Check: http://{server_config['host']}:{server_config['port']}/health")
    print(f"⚠️  Note: This is demo mode. Train models for full functionality.")
    
    uvicorn.run(
        "app_demo:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config['reload']
    )