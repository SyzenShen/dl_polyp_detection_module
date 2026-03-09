import os
from ultralytics import YOLO
from django.conf import settings

class PolypDetector:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Load model only once
        # Default to yolo11n.pt in root
        model_path = os.path.join(settings.BASE_DIR, 'yolo11n.pt')
        
        # Check if trained model exists
        best_model_path = os.path.join(settings.BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
             print(f"Found trained model at {best_model_path}")
             model_path = best_model_path
        
        print(f"Loading YOLO model from {model_path}...")
        try:
            self._model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback or re-raise?
            # For now, let's assume yolo11n.pt is always there or downloadable
            self._model = YOLO('yolo11n.pt')

    def predict(self, image_path):
        # Run inference
        if not self._model:
            return []
            
        # Lower confidence threshold to 0.05 for demo purposes
        # Enable NMS with iou=0.4 to reduce overlapping boxes
        results = self._model(image_path, conf=0.05, iou=0.4)
        
        detections = []
        for result in results:
            # result.boxes contains bounding boxes
            for box in result.boxes:
                # box.xyxy[0] is [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                detections.append({
                    "bbox": coords,
                    "confidence": conf,
                    "label": label
                })
        return detections
