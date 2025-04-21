"""
YOLO Object Detector implementation.
This module provides the YOLODetector class for detecting objects in images.
"""

import numpy as np
import torch
from ultralytics import YOLO

class YOLODetector:
    """
    YOLOv8 based object detector.
    
    Attributes:
        model: The YOLO model instance
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        classes: List of class indices to detect (None for all classes)
    """
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.45, 
                 iou_threshold=0.45, classes=None):
        """
        Initialize the YOLODetector.
        
        Args:
            model_name: Name or path of the YOLO model to use
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            classes: List of class names to detect (None for all classes)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load the YOLO model
        try:
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get class names from model
        self.class_names = self.model.names
        
        # Convert class names to class indices if provided
        self.classes = None
        if classes:
            self.classes = []
            for class_name in classes:
                for idx, name in self.class_names.items():
                    if name.lower() == class_name.lower():
                        self.classes.append(idx)
                        break
            
            if not self.classes:
                print("Warning: None of the specified classes found in model")
                self.classes = None
    
    def detect(self, frame):
        """
        Perform object detection on a frame.
        
        Args:
            frame: The input image frame (numpy array)
            
        Returns:
            List of detection results, each in format:
            [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Perform detection
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )
        
        # Process results
        detections = []
        
        if len(results) > 0:
            result = results[0]  # First image result
            
            # Extract detection information
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Get bbox in xyxy format
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                # Format: [x1, y1, x2, y2, confidence, class_id, class_name]
                detection = [*xyxy, conf, cls_id, cls_name]
                detections.append(detection)
        
        return detections
    
    def get_class_name(self, class_id):
        """Get class name from class ID."""
        return self.class_names.get(class_id, "unknown")
