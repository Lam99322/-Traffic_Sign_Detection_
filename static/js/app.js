// Traffic Sign Detection Web Application JavaScript

class TrafficSignDetector {
    constructor() {
        this.selectedFile = null;
        this.initializeElements();
        this.bindEvents();
        this.loadClasses();
    }

    initializeElements() {
        this.elements = {
            uploadArea: document.getElementById('uploadArea'),
            fileInput: document.getElementById('fileInput'),
            uploadBtn: document.getElementById('uploadBtn'),
            clearBtn: document.getElementById('clearBtn'),
            preview: document.getElementById('preview'),
            results: document.getElementById('results'),
            loading: document.getElementById('loading'),
            stats: document.getElementById('stats'),
            totalDetections: document.getElementById('totalDetections'),
            avgConfidence: document.getElementById('avgConfidence'),
            processingTime: document.getElementById('processingTime')
        };
    }

    bindEvents() {
        // Click to upload
        this.elements.uploadArea.addEventListener('click', () => this.elements.fileInput.click());

        // File selection
        this.elements.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this.handleFile(file);
        });

        // Drag and drop
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });

        this.elements.uploadArea.addEventListener('dragleave', () => {
            this.elements.uploadArea.classList.remove('dragover');
        });

        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleFile(file);
            }
        });

        // Buttons
        this.elements.uploadBtn.addEventListener('click', () => this.analyzeImage());
        this.elements.clearBtn.addEventListener('click', () => this.clearResults());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.clearResults();
            if (e.key === 'Enter' && this.selectedFile) this.analyzeImage();
        });
    }

    async loadClasses() {
        try {
            const response = await fetch('/api/classes');
            const data = await response.json();
            if (data.success) {
                this.classes = data.classes;
                console.log('Loaded classes:', this.classes);
            }
        } catch (error) {
            console.error('Failed to load classes:', error);
        }
    }

    handleFile(file) {
        // Validate file
        if (!this.validateFile(file)) return;

        this.selectedFile = file;
        
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.elements.preview.innerHTML = `
                <img id="previewImage" src="${e.target.result}" alt="Preview" class="fade-in">
            `;
        };
        reader.readAsDataURL(file);
        
        this.elements.uploadBtn.disabled = false;
        this.elements.results.innerHTML = '';
        this.elements.stats.style.display = 'none';
        
        this.showNotification('Ảnh đã được tải lên thành công!', 'success');
    }

    validateFile(file) {
        const maxSize = 5 * 1024 * 1024; // 5MB
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        
        if (!allowedTypes.includes(file.type)) {
            this.showNotification('Chỉ h hỗ trợ file JPG, JPEG, PNG', 'error');
            return false;
        }
        
        if (file.size > maxSize) {
            this.showNotification('Kích thước file không được vượt quá 5MB', 'error');
            return false;
        }
        
        return true;
    }

    async analyzeImage() {
        if (!this.selectedFile) return;

        const formData = new FormData();
        formData.append('file', this.selectedFile);

        this.showLoading(true);
        this.elements.results.innerHTML = '';
        this.elements.stats.style.display = 'none';

        const startTime = performance.now();

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const endTime = performance.now();
            const processingTime = Math.round(endTime - startTime);

            if (data.success) {
                this.displayResults(data, processingTime);
                this.showNotification('Phân tích hoàn tất!', 'success');
            } else {
                throw new Error(data.detail || 'Không thể xử lý ảnh');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.displayError(error.message);
            this.showNotification('Có lỗi xảy ra khi phân tích', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(data, processingTime) {
        const { num_detections, detections, image_size } = data;

        if (num_detections === 0) {
            this.elements.results.innerHTML = `
                <div class="detection-item slide-in">
                    <h3>📭 Không tìm thấy biển báo nào</h3>
                    <p>Hãy thử với ảnh khác có biển báo rõ ràng hơn.</p>
                </div>
            `;
            return;
        }

        // Calculate statistics
        const avgConf = detections.reduce((sum, d) => sum + d.confidence, 0) / num_detections;

        // Display statistics
        this.elements.stats.style.display = 'flex';
        this.elements.totalDetections.textContent = num_detections;
        this.elements.avgConfidence.textContent = `${(avgConf * 100).toFixed(1)}%`;
        this.elements.processingTime.textContent = `${processingTime}ms`;

        // Display detections
        let html = '<h2 style="margin-bottom: 20px;">📋 Kết quả phát hiện:</h2>';
        
        detections.forEach((det, idx) => {
            const [x1, y1, x2, y2] = det.bbox;
            const conf = (det.confidence * 100).toFixed(1);
            const classInfo = this.classes ? ` (${this.classes[det.class_id]})` : '';
            
            html += `
                <div class="detection-item slide-in" style="animation-delay: ${idx * 0.1}s">
                    <h3>🚦 Biển báo ${idx + 1}: ${det.class_name}${classInfo}</h3>
                    <span class="confidence">Độ tin cậy: ${conf}%</span>
                    <span class="bbox">Vị trí: [${x1}, ${y1}, ${x2}, ${y2}]</span>
                    <div style="margin-top: 10px;">
                        <strong>Kích thước:</strong> ${x2 - x1} × ${y2 - y1} pixels
                    </div>
                    ${det.cnn_prediction ? `
                        <div class="cnn-prediction">
                            <strong>🎯 CNN xác nhận:</strong> 
                            Class ${det.cnn_prediction.class_id} 
                            (${(det.cnn_prediction.confidence * 100).toFixed(1)}%)
                        </div>
                    ` : ''}
                </div>
            `;
        });

        // Add image info
        html += `
            <div class="detection-item" style="border-left-color: #17a2b8;">
                <h3>📊 Thông tin ảnh</h3>
                <p><strong>Kích thước:</strong> ${image_size.width} × ${image_size.height} pixels</p>
                <p><strong>Thời gian xử lý:</strong> ${processingTime}ms</p>
                <p><strong>Tổng số biển báo:</strong> ${num_detections}</p>
            </div>
        `;

        this.elements.results.innerHTML = html;
    }

    displayError(message) {
        this.elements.results.innerHTML = `
            <div class="detection-item slide-in" style="border-left-color: var(--danger-color);">
                <h3>❌ Lỗi</h3>
                <p>${message}</p>
                <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    Vui lòng thử lại hoặc liên hệ quản trị viên.
                </p>
            </div>
        `;
    }

    showLoading(show) {
        this.elements.loading.style.display = show ? 'block' : 'none';
        this.elements.uploadBtn.disabled = show;
        
        if (show) {
            this.elements.uploadBtn.innerHTML = '⏳ Đang xử lý...';
        } else {
            this.elements.uploadBtn.innerHTML = '📤 Tải lên và Phân tích';
        }
    }

    clearResults() {
        this.selectedFile = null;
        this.elements.preview.innerHTML = '';
        this.elements.results.innerHTML = '';
        this.elements.stats.style.display = 'none';
        this.elements.fileInput.value = '';
        this.elements.uploadBtn.disabled = true;
        
        this.showNotification('Đã xóa kết quả', 'info');
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notif => notif.remove());

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Batch processing (future enhancement)
    async processBatch(files) {
        if (files.length > 10) {
            this.showNotification('Tối đa 10 ảnh mỗi lần', 'error');
            return;
        }

        const results = [];
        for (const file of files) {
            const result = await this.processSingleFile(file);
            results.push(result);
        }
        
        return results;
    }

    async processSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TrafficSignDetector();
    console.log('🚦 Traffic Sign Detection System initialized');
});

// Utility functions
const Utils = {
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },

    formatTime(ms) {
        if (ms < 1000) return `${ms}ms`;
        return `${(ms / 1000).toFixed(2)}s`;
    },

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};