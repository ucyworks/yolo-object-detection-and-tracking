"""
Visualization utilities for object detection and tracking.
This module provides functions for drawing bounding boxes, labels, and other visual elements.
"""

import cv2
import numpy as np
import random

class Visualizer:
    """Class for visualizing detection and tracking results."""
    
    def __init__(self, display_config):
        """
        Initialize the visualizer with display configuration.
        
        Args:
            display_config: Dictionary containing display settings
        """
        self.config = display_config
        self.colors = {}  # Cache for class colors
        self.track_colors = {}  # Cache for track colors
    
    def _get_color(self, class_id, track_id=None):
        """Get a consistent color for a class or track ID."""
        if track_id is not None:
            # Use track ID for color if tracking
            if track_id not in self.track_colors:
                self.track_colors[track_id] = [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ]
            return self.track_colors[track_id]
        else:
            # Use class ID for color if just detecting
            if class_id not in self.colors:
                self.colors[class_id] = [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ]
            return self.colors[class_id]
    
    def draw_box(self, img, bbox, color, thickness):
        """Draw a bounding box on the image."""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img
    
    def draw_label(self, img, bbox, text, color):
        """Draw a label with text above the bounding box."""
        x1, y1, _, _ = map(int, bbox)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config['text_scale']
        font_thickness = self.config['text_thickness']
        
        # Get text size and calculate background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img, 
            (x1, y1 - text_height - 8), 
            (x1 + text_width + 8, y1),
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            img, 
            text, 
            (x1 + 4, y1 - 4), 
            font, 
            font_scale, 
            (0, 0, 0),  # Black text
            font_thickness,
            cv2.LINE_AA
        )
        
        return img
    
    def draw_results(self, img, results):
        """
        Draw detection/tracking results on the image.
        
        Args:
            img: The input image
            results: List of results, each in format
                   [x1, y1, x2, y2, confidence, class_id, class_name, track_id (optional)]
                   
        Returns:
            Image with visualizations added
        """
        # Make a copy to avoid modifying the original
        output_img = img.copy()
        
        for result in results:
            bbox = result[:4]
            confidence = result[4]
            class_id = result[5]
            class_name = result[6]
            
            # Get track ID if available (for tracking)
            track_id = None
            if len(result) > 7:
                track_id = result[7]
            
            # Get appropriate color
            color = self._get_color(class_id, track_id)
            
            # Draw bounding box
            if self.config['show_boxes']:
                output_img = self.draw_box(
                    output_img, 
                    bbox, 
                    color, 
                    self.config['box_thickness']
                )
            
            # Prepare label text
            label_text = ""
            if self.config['show_labels']:
                label_text += class_name
            
            if self.config['show_confidence']:
                if label_text:
                    label_text += f": {confidence:.2f}"
                else:
                    label_text += f"{confidence:.2f}"
            
            # Add track ID if available
            if track_id is not None:
                if label_text:
                    label_text += f" ID:{track_id}"
                else:
                    label_text += f"ID:{track_id}"
            
            # Draw label if needed
            if label_text and (self.config['show_labels'] or self.config['show_confidence']):
                output_img = self.draw_label(output_img, bbox, label_text, color)
        
        return output_img
    
    def draw_fps(self, img, fps):
        """Draw FPS counter on the image."""
        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = tuple(self.config['text_color'])
        
        # Draw text with background
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        cv2.rectangle(
            img, 
            (10, 10), 
            (10 + text_width + 10, 10 + text_height + 10),
            (0, 0, 0), 
            -1
        )
        
        cv2.putText(
            img, 
            text, 
            (15, 30), 
            font, 
            font_scale, 
            color, 
            thickness,
            cv2.LINE_AA
        )
        
        return img
