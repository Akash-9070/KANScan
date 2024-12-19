# KAN Tumor Detection

A deep learning-based tumor detection system using Kolmogorov-Arnold Networks (KANs) for analyzing CT scan images. This system provides accurate tumor detection with bounding box localization.

## Features

- Implementation of Kolmogorov-Arnold Networks for medical image analysis
- Tumor detection with bounding box prediction
- Support for CT scan image processing
- Real-time visualization of detection results
- Comprehensive training pipeline
- Model evaluation tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Akash-9070/KANScan.git
cd kan-tumor-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations.txt
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations.txt
└── test/
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Annotation format (CSV):
```
image_name,x,y,width,height,label
image1.jpg,100,150,50,60,1
image2.jpg,200,300,80,90,1
```

## Usage

### Training

1. Update the paths in `main()` to point to your dataset:
```python
train_dataset = CTScanDataset(
    image_dir='path/to/train/images',
    annotation_file='path/to/train/annotations.txt',
    transform=transform
)
```

2. Run the training script:
```bash
python tumor_detection.py
```

### Inference

```python
from tumor_detection import KANetwork, detect_tumor

# Load model
model = KANetwork()
model.load_state_dict(torch.load('tumor_detection_kan.pth'))

# Detect tumor
bbox, confidence = detect_tumor(model, 'path/to/image.jpg')
```

## Model Architecture

The KAN tumor detection model consists of:

1. Feature Extraction Layer:
   - Convolutional neural network for initial feature extraction
   - Adaptive pooling for fixed-size feature maps

2. Kolmogorov-Arnold Network:
   - Multiple KolmogorovLayers implementing the universal approximation theorem
   - Inner (Ψ) and outer (g) functions for complex pattern recognition
   - Hierarchical feature representation

3. Detection Heads:
   - Bounding box regression for tumor localization
   - Classification head for tumor presence prediction

## Training Details

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Functions: 
  - SmoothL1Loss for bounding box regression
  - BCELoss for classification
- Batch Size: 16
- Image Size: 256x256
- Training Epochs: 50

## Results Visualization

The model provides:
- Bounding box coordinates (x, y, width, height)
- Confidence score for tumor detection
- Real-time visualization with matplotlib
- Color-coded boxes based on confidence levels

## Evaluation Metrics

The model is evaluated using:
- Mean Average Precision (mAP)
- Intersection over Union (IoU)
- Classification accuracy
- Recall and precision metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kan_tumor_detection,
  author = {[Akash]},
  title = {KAN Tumor Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Akash-9070/KANScan.git}
}
```

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/Akash-9070/KANScan.git](https://github.com/Akash-9070/KANScan.git)
