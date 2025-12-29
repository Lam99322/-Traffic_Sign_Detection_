🚦 Traffic Sign Detection using YOLOv11
📌 Giới thiệu

Dự án Traffic Sign Detection được xây dựng nhằm mục tiêu nhận diện và phân loại biển báo giao thông trong môi trường thực tế tại Việt Nam.
Hệ thống sử dụng YOLOv11 – mô hình học sâu hiện đại cho bài toán Object Detection, đảm bảo tốc độ nhanh, độ chính xác cao và khả năng triển khai thực tế.

Dự án phù hợp cho:

Đồ án học phần / khóa luận

Nghiên cứu thị giác máy tính

Ứng dụng giao thông thông minh (ITS)

🎯 Mục tiêu đề tài

Nhận diện các loại biển báo giao thông Việt Nam từ hình ảnh và video

Huấn luyện mô hình YOLOv11 với dataset tùy chỉnh

Đánh giá hiệu năng mô hình (Precision, Recall, mAP)

Sẵn sàng triển khai thực tế

🧠 Công nghệ sử dụng

Python 3.9+

YOLOv11 (Ultralytics)

PyTorch

OpenCV

NumPy

Matplotlib

Docker (tùy chọn triển khai)

📁 Cấu trúc thư mục
Traffic_Sign_Detection/
│
├── data/                   # Dataset (images, labels)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                 # Trọng số mô hình
│   └── best.pt
│
├── src/                    # Source code
│   ├── train.py
│   ├── detect.py
│   ├── validate.py
│   └── utils.py
│
├── advanced_training.py    # Pipeline huấn luyện nâng cao
├── requirements.txt        # Thư viện cần thiết
├── Dockerfile              # Cấu hình Docker
└── README.md

⚙️ Cài đặt môi trường
1️⃣ Clone project
git clone https://github.com/Lam99322/Traffic_Sign_Detection.git
cd Traffic_Sign_Detection

2️⃣ Cài đặt thư viện
pip install -r requirements.txt

🚀 Huấn luyện mô hình
Huấn luyện nhanh (50 epochs)
python advanced_training.py


Hoặc huấn luyện thủ công:

python src/train.py

🔍 Kiểm tra & nhận diện
python src/detect.py --source data/test

📊 Đánh giá mô hình
