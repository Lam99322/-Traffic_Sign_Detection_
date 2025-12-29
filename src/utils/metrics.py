import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, class_names=None):
        """Tính toán các metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Classification report
        if class_names:
            report = classification_report(y_true, y_pred, 
                                          target_names=class_names,
                                          output_dict=True)
            metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    @staticmethod
    def calculate_detection_metrics(pred_boxes, true_boxes, iou_threshold=0.5):
        """Calculate object detection metrics"""
        
        def calculate_iou(box1, box2):
            # box format: [x1, y1, x2, y2]
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        # Simple matching: for each true box, find best pred with IoU > threshold
        matched_ious = []
        used_preds = set()
        
        for true_box in true_boxes:
            best_iou = iou_threshold
            best_pred_idx = -1
            
            for i, pred_box in enumerate(pred_boxes):
                if i in used_preds:
                    continue
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            if best_pred_idx != -1:
                matched_ious.append(best_iou)
                used_preds.add(best_pred_idx)
        
        tp = len(matched_ious)
        fp = len(pred_boxes) - tp
        fn = len(true_boxes) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        mean_iou = np.mean(matched_ious) if matched_ious else 0
        
        return {
            'mean_iou': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }