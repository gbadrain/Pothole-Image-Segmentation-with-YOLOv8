# Pothole Image Segmentation with YOLOv8

A deep learning solution for automated pothole detection and segmentation using YOLOv8, enabling precise road condition assessment and infrastructure monitoring.

## Project Overview

This project implements a state-of-the-art computer vision system that combines YOLOv8's object detection capabilities with instance segmentation to identify potholes in road imagery. The model not only detects pothole locations but also generates pixel-accurate segmentation masks, providing detailed geometric information for road maintenance planning.

## Project Structure

```
pothole-segmentation/
├── data/                      # Dataset and configuration files
│   ├── data.yaml             # YOLOv8 dataset configuration
│   ├── train/                # Training dataset
│   │   ├── images/           # Training images
│   │   └── labels/           # YOLO format annotations
│   ├── valid/                # Validation dataset
│   ├── valid-mini/           # Minimal validation subset
│   └── test/                 # Test images and annotations
├── runs/                      # Training outputs and experiments
│   └── segment/
│       └── train2/           # Latest training run
│           ├── weights/      # Model checkpoints
│           │   ├── best.pt   # Best performing weights
│           │   └── last.pt   # Final epoch weights
│           ├── results.png   # Training metrics visualization
│           └── val_batch0_pred.jpg  # Validation predictions
├── src/                       # Source code and utilities
│   ├── convert_annotations.py      # Annotation format converter
│   ├── custom_coco_to_yolo.py     # COCO to YOLO converter
│   └── train_model.py              # Model training script
├── models/                    # Pre-trained and custom models
│   ├── yolov8n-seg.pt        # Base YOLOv8 nano segmentation model
│   └── best.pt               # Trained model (symlink)
├── docs/                      # Documentation
│   ├── PROJECT_LOG.md        # Development log and iterations
│   └── POT_SHOT_EPOCH.pdf    # Training epoch analysis report
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Key Features

- **Precise Detection**: YOLOv8-powered pothole detection with high accuracy
- **Instance Segmentation**: Pixel-level segmentation masks for detailed analysis
- **Comprehensive Metrics**: Training loss, mAP, precision, and recall tracking
- **Custom Tools**: Annotation conversion and data preprocessing utilities
- **Fast Inference**: Optimized for real-time road condition assessment
- **Deployment Ready**: Easy integration for web, mobile, or edge applications

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gbadrain/pothole-segmentation.git
cd pothole-segmentation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Single Image Inference

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/segment/train2/weights/best.pt')

# Run inference
results = model('path/to/your/road_image.jpg')

# Visualize results
for result in results:
    # Display annotated image
    annotated_image = result.plot()
    cv2.imshow('Pothole Detection', annotated_image)
    cv2.waitKey(0)
    
    # Save results
    result.save('output/detection_result.jpg')
```

#### Batch Processing

```bash
# Process multiple images
yolo predict model=runs/segment/train2/weights/best.pt \
             source='data/test/images/' \
             save=True \
             conf=0.5 \
             save_txt=True
```

#### Real-time Video Processing

```python
from ultralytics import YOLO

model = YOLO('runs/segment/train2/weights/best.pt')

# Process video stream
results = model.track(source='road_video.mp4', 
                     show=True, 
                     save=True,
                     tracker='bytetrack.yaml')
```

## Training Your Own Model

### Dataset Preparation

1. **Organize your dataset** following the YOLO format:
```
your_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. **Convert annotations** (if using COCO format):
```bash
python src/custom_coco_to_yolo.py \
    --input data/annotations.json \
    --output_dir data/labels \
    --class_names pothole
```

3. **Update data.yaml** configuration:
```yaml
path: ./data
train: train/images
val: valid/images
test: test/images

nc: 1  # number of classes
names: ['pothole']
```

### Training Process

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n-seg.pt')  # or yolov8s-seg.pt, yolov8m-seg.pt

# Train the model
results = model.train(
    data='data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='pothole_segmentation_v2',
    patience=20,
    save_period=10,
    val=True,
    plots=True
)

# Evaluate model
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## Performance Metrics

Our trained model achieves the following performance on the validation set:

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP@0.5** | 0.892 | Mean Average Precision at IoU 0.5 |
| **Precision** | 0.856 | Detection precision (true positives / total detections) |
| **Recall** | 0.831 | Detection recall (true positives / total ground truth) |
| **Mask Precision** | 0.812 | Segmentation mask precision |
| **Mask Recall** | 0.794 | Segmentation mask recall |
| **Inference Speed** | ~45ms | Average inference time per image (GPU) |

## Advanced Configuration

### Custom Training Parameters

```python
# Advanced training configuration
results = model.train(
    data='data/data.yaml',
    epochs=200,
    imgsz=832,                    # Higher resolution for better accuracy
    batch=8,                      # Adjust based on GPU memory
    lr0=0.001,                   # Initial learning rate
    weight_decay=0.0005,         # L2 regularization
    mosaic=1.0,                  # Mosaic augmentation probability
    mixup=0.1,                   # MixUp augmentation probability
    copy_paste=0.3,              # Copy-paste augmentation
    degrees=10.0,                # Rotation augmentation
    translate=0.2,               # Translation augmentation
    scale=0.9,                   # Scale augmentation
    fliplr=0.5,                  # Horizontal flip probability
    hsv_h=0.015,                 # HSV hue augmentation
    hsv_s=0.7,                   # HSV saturation augmentation
    hsv_v=0.4,                   # HSV value augmentation
)
```

### Model Export Options

```python
# Export to different formats
model = YOLO('runs/segment/train2/weights/best.pt')

# ONNX format (recommended for deployment)
model.export(format='onnx', dynamic=True)

# TensorRT (NVIDIA GPUs)
model.export(format='engine', device=0)

# CoreML (Apple devices)
model.export(format='coreml')

# TensorFlow Lite (mobile deployment)
model.export(format='tflite')
```

## Deployment Options

### Web API Deployment

```python
from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import io
from PIL import Image

app = Flask(__name__)
model = YOLO('runs/segment/train2/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect_potholes():
    # Get image from request
    image_data = request.json['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Run inference
    results = model(image)
    
    # Process results
    detections = []
    for result in results:
        for box, mask in zip(result.boxes, result.masks):
            detections.append({
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist(),
                'mask': mask.data.tolist()
            })
    
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
FROM ultralytics/ultralytics:latest

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "api.py"]
```

## Model Versioning and Monitoring

Track model performance across different versions:

```python
import mlflow
import mlflow.pytorch

# Log training metrics
with mlflow.start_run():
    mlflow.log_params({
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_size': 'yolov8n'
    })
    
    mlflow.log_metrics({
        'mAP50': 0.892,
        'precision': 0.856,
        'recall': 0.831
    })
    
    mlflow.pytorch.log_model(model, 'pothole_detector')
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Ultralytics YOLOv8
- OpenCV 4.5+
- NumPy
- Pillow
- Matplotlib

See `requirements.txt` for complete dependency list.

## Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Reduce batch size
python train.py --batch 8  # instead of 16

# Or use gradient accumulation
python train.py --batch 4 --accumulate 4
```

**Low mAP Performance**
- Increase training epochs
- Use higher resolution images (`imgsz=832`)
- Add more training data
- Adjust augmentation parameters

**Slow Inference**
- Use smaller model variant (`yolov8n-seg.pt`)
- Reduce input image size
- Enable GPU acceleration
- Consider model quantization

## Future Development

### Planned Features

- **Multi-class Detection**: Extend to detect cracks, road markings, and other road defects
- **Severity Classification**: Implement pothole severity scoring (minor, moderate, severe)
- **Size Estimation**: Calculate pothole dimensions from segmentation masks
- **GIS Integration**: GPS coordinate mapping for infrastructure databases
- **Mobile App**: Real-time detection on smartphones
- **Dashboard**: Web-based monitoring and analytics platform

### Research Directions

- Integration with LiDAR data for 3D analysis
- Temporal analysis for pothole progression tracking
- Weather condition impact assessment
- Cost estimation models for repair prioritization


## Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **COCO Dataset** for annotation format standards  
- **OpenCV Community** for computer vision tools
- **Copilot AI** for development assistance
- **Gemini CLI** for development assistance
- **Contributors** who provided dataset annotations and testing

## Contact

**Gurpreet Singh Badrain** - *Data Analyst & Market Research Analyst*

- **Portfolio**: [Data Guru](https://data-guru-portfolio.com)
- **LinkedIn**: [gurpreet-badrain](https://linkedin.com/in/gurpreet-badrain)
- **Email**: [gbadrain@gmail.com](mailto:gbadrain@gmail.com)
- **GitHub**: [gbadrain](https://github.com/gbadrain)
- **Streamlit**: [gbadrain-Machine Learning](https://streamlit.com/gbadrain-machine-learning)

---

<div align="center">

**Star this repository if you found it helpful!**

[Back to Top](#pothole-image-segmentation-with-yolov8)

</div>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
