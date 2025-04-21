"""
Data handling utilities for the YOLO object detection and tracking application.
This module provides utilities for loading configurations and setting up data sources.
"""

import os
import yaml
import cv2

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        raise

def find_available_camera():
    """
    Find the first available camera by testing multiple indices.
    
    Returns:
        Camera index if found, None otherwise
    """
    # Try common camera indices
    for index in [0, 1, 2, -1]:  # -1 is a special index that sometimes works
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    return index
        except Exception:
            continue
    
    # If we get here, try a more extensive search
    for index in range(10):
        if index in [0, 1, 2]:  # Already tested
            continue
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None and frame.size > 0:
                    return index
        except Exception:
            continue
    
    return None

def setup_camera(source, config):
    """
    Setup camera or video source.
    
    Args:
        source: Camera index or video file path
        config: Configuration dictionary containing video settings
        
    Returns:
        Tuple of (VideoCapture object, frame width, frame height)
    """
    # Determine if source is a file or camera index
    try:
        source = int(source)  # Camera index
    except ValueError:
        # Assume it's a file path
        if not os.path.exists(source):
            print(f"Warning: Source file {source} does not exist.")
    
    # Open the video source
    cap = cv2.VideoCapture(source)
    
    # Set camera properties if applicable
    video_config = config.get('video', {})
    if isinstance(source, int):  # Only set these for camera sources
        if 'width' in video_config:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config['width'])
        if 'height' in video_config:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config['height'])
        if 'fps' in video_config:
            cap.set(cv2.CAP_PROP_FPS, video_config['fps'])
    
    # Get the actual frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, frame_width, frame_height

def create_dirs():
    """Create necessary directories for the application."""
    dirs = ['output', 'logs', 'config']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
