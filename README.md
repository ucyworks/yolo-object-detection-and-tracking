# YOLO Object Detection and Tracking

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)

A robust, modular object detection and tracking system built with YOLOv8, OpenCV, and PyTorch. This project provides real-time object detection and tracking capabilities from a camera feed.

## Features

- **Real-time Object Detection**: Utilizes YOLOv8 for state-of-the-art object detection with 80+ classes
- **Object Tracking**: Implements advanced tracking algorithms to maintain object identity across frames
- **Flexible Configuration**: Easily configurable through YAML files for different deployment scenarios
- **Robust Visualization**: Real-time annotated video feed with bounding boxes, labels, and confidence scores
- **Performance Optimized**: Multi-threading support for optimal CPU/GPU resource usage
- **Extensible Architecture**: Modular design allows for easy integration of new models or tracking algorithms

## Installation

```bash
# Clone the repository
git clone https://github.com/ucytv/yolo-object-detection.git/
cd yolo-object-detection

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 weights
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Basic Usage

```bash
# Run with default camera
python main.py

# Run with specific camera index
python main.py --source 1

# Run with a video file
python main.py --source path/to/video.mp4
```

### Advanced Configuration

Edit the configuration file at `config/yolo_config.yaml` to customize:
- Detection confidence threshold
- Object classes to detect
- Tracking algorithm parameters
- Display options

## Project Structure

```
├── main.py                 # Main application entry point
├── models/                 # Model implementations
│   ├── detector.py         # YOLO object detector
│   └── tracker.py          # Multiple object tracker
├── utils/                  # Utility functions
│   ├── visualization.py    # Visualization tools
│   └── data_utils.py       # Data handling utilities
└── config/                 # Configuration files
    └── yolo_config.yaml    # YOLO & tracking parameters
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for tracking algorithms
- [OpenCV](https://opencv.org/) for computer vision utilities