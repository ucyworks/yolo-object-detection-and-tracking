#!/usr/bin/env python3
"""
Main entry point for the YOLO Object Detection and Tracking application.
Handles command line arguments, initializes components, and runs the main loop.
"""

import argparse
import sys
import cv2
import yaml
import time
from pathlib import Path

from models.detector import YOLODetector
from models.tracker import ObjectTracker
from utils.visualization import Visualizer
from utils.data_utils import load_config, setup_camera, find_available_camera

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLO Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='auto',
                        help='Source for detection: "auto" to find available camera, webcam index, or video file path')
    parser.add_argument('--config', type=str, default='config/yolo_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--show', action='store_true', help='Display the processed frames')
    parser.add_argument('--save', action='store_true', help='Save the processed video')
    parser.add_argument('--camera-test', action='store_true', help='Test available cameras and exit')
    return parser.parse_args()

def main():
    """Main function to run the object detection and tracking pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Test available cameras if requested
    if args.camera_test:
        print("Testing available cameras...")
        available_cameras = []
        for i in range(10):  # Test camera indices 0-9
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera index {i} is available")
                    available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            print(f"Available camera indices: {available_cameras}")
            print(f"Use with: --source {available_cameras[0]}")
        else:
            print("No cameras found. Try connecting a webcam or check permissions.")
        return
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.save:
        config['save']['enabled'] = True
    
    # Initialize camera or video source
    try:
        # Handle automatic camera detection
        if args.source == 'auto':
            camera_index = find_available_camera()
            if camera_index is None:
                print("Error: No available cameras found.")
                print("Try connecting a webcam, checking permissions, or specify a video file with --source path/to/video.mp4")
                sys.exit(1)
            print(f"Using automatically detected camera at index {camera_index}")
            source = camera_index
        else:
            source = args.source
        
        cap, frame_width, frame_height = setup_camera(source, config)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            if isinstance(source, int) or source.isdigit():
                print("Tips for fixing camera access issues:")
                print("1. Try running with --camera-test to find available cameras")
                print("2. Check if another application is using the camera")
                print("3. Try a different camera index (--source 1 or --source 2)")
                print("4. Check camera permissions for your application")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error setting up camera: {e}")
        print("Try running with --camera-test to find available cameras")
        sys.exit(1)
    
    # Initialize detector, tracker and visualizer
    detector = YOLODetector(
        model_name=config['model']['name'],
        conf_threshold=config['model']['confidence'],
        iou_threshold=config['model']['iou_threshold'],
        classes=config.get('classes', None)
    )
    
    tracker = ObjectTracker(
        tracker_type=config['tracker']['type'],
        max_age=config['tracker']['max_age'],
        min_hits=config['tracker']['min_hits'],
        iou_threshold=config['tracker']['iou_threshold']
    )
    
    visualizer = Visualizer(config['display'])
    
    # Setup video writer if saving is enabled
    out = None
    if config['save']['enabled']:
        fourcc = cv2.VideoWriter_fourcc(*config['save']['codec'])
        output_path = Path(config['save']['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            config['video']['fps'], 
            (frame_width, frame_height)
        )
    
    print("Starting object detection and tracking...")
    try:
        while True:
            # Read frame from camera
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = detector.detect(frame)
            
            # Track objects
            tracks = tracker.update(detections)
            
            # Visualize results
            output_frame = visualizer.draw_results(frame, tracks)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            if config['display']['show_fps']:
                output_frame = visualizer.draw_fps(output_frame, fps)
            
            # Display the resulting frame
            if args.show:
                cv2.imshow('YOLO Object Detection & Tracking', output_frame)
            
            # Save the frame
            if out is not None:
                out.write(output_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    main()
