# Deepfake Detection System 🎭

A deep learning-based system for detecting deepfake videos using transfer learning with EfficientNet-B0. This project implements a complete pipeline from video frame extraction to model training and evaluation.

## 🎯 Project Overview

This system analyzes video frames to classify videos as either **real** or **fake (deepfake)** with high accuracy. It uses state-of-the-art deep learning techniques and transfer learning to identify subtle artifacts and inconsistencies that are characteristic of deepfake content.

## ✨ Key Features

- **Automated Frame Extraction**: Intelligent video processing with configurable sampling rates
- **Transfer Learning**: Leverages pre-trained EfficientNet-B0 for superior performance
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and visualizations
- **Real-time Predictions**: Function to classify new videos on-the-fly
- **Flexible Configuration**: Easily adjustable hyperparameters
- **GPU Acceleration**: CUDA support for faster training

## 📊 Dataset Structure

The project expects the following directory structure:

```
your_project_folder/
├── Celeb-real/              # Real celebrity videos
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── Celeb-synthesis/         # Deepfake/synthetic videos
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── YouTube-real/            # Real YouTube videos
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── project.py               # Main script
```

**Supported formats**: `.mp4`, `.avi`, `.mov`

## 🛠️ Technical Stack

### Core Libraries
- **PyTorch**: Deep learning framework
- **TorchVision**: Pre-trained models and transforms
- **OpenCV**: Video processing and frame extraction
- **NumPy**: Numerical computations
- **Scikit-learn**: Model evaluation and metrics

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualizations

### Additional Tools
- **tqdm**: Progress bars
- **Pillow**: Image processing

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd deepfake-detection
```

2. **Install dependencies**:
```bash
pip install torch torchvision opencv-python numpy scikit-learn matplotlib seaborn tqdm pillow
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

3. **Verify CUDA installation** (optional, for GPU acceleration):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Usage

### Basic Usage

1. **Prepare your dataset** following the structure above

2. **Run the script**:
```bash
python project.py
```

### Configuration

Adjust parameters in the `CONFIG` dictionary:

```python
CONFIG = {
    'base_path': '.',                    # Dataset directory
    'frame_extraction_rate': 10,         # Extract 1 frame every N frames
    'max_frames_per_video': 20,          # Maximum frames per video
    'max_videos_per_folder': 50,         # Limit videos (None for all)
    'img_size': 224,                     # Image size for model
    'batch_size': 16,                    # Training batch size
    'epochs': 10,                        # Training epochs
    'learning_rate': 0.0001,             # Learning rate
}
```

### Making Predictions on New Videos

```python
# Load the trained model
model.load_state_dict(torch.load('best_model.pth'))

# Predict on a new video
prediction, confidence = predict_video(
    'path/to/video.mp4', 
    model, 
    test_transform, 
    device
)

print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
print(f"Confidence: {confidence:.2%}")
```

## 🧠 Model Architecture

### Base Model: EfficientNet-B0
- Pre-trained on ImageNet
- Transfer learning approach
- Custom classifier head:
  - Dropout (p=0.3)
  - Fully connected layer (1280 → 2 classes)

### Training Strategy
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: CrossEntropyLoss
- **Data Split**: 80% train, 20% test
- **Validation**: 20% of training data
- **Early Stopping**: Best model saved based on validation accuracy

## 📈 Model Pipeline

```
Video Input
    ↓
Frame Extraction (every 10th frame)
    ↓
Preprocessing (224x224, normalization)
    ↓
Data Augmentation (training only)
    ↓
EfficientNet-B0 Feature Extraction
    ↓
Custom Classifier
    ↓
Prediction (Real/Fake)
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation including:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for real and fake classes
- **Recall**: Recall for real and fake classes
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **Training Curves**: Loss and accuracy over epochs

## 📁 Output Files

After training, the following files are generated:

1. **best_model.pth** - Model with highest validation accuracy
2. **deepfake_detector_final.pth** - Final trained model
3. **training_history.png** - Training and validation curves
4. **confusion_matrix.png** - Confusion matrix visualization
5. **sample_predictions.png** - Sample prediction results
6. **model_config.json** - Configuration parameters

## 🎯 Performance

Typical performance metrics (depends on dataset quality and size):

- **Validation Accuracy**: 85-95%
- **Test Accuracy**: 80-90%
- **Training Time**: ~10-30 minutes (10 epochs, GPU)

*Note: Actual performance varies based on dataset quality, size, and diversity.*

## 🔧 Advanced Features

### Data Augmentation
- Random horizontal flip
- Color jitter (brightness, contrast)
- Random rotation (±10°)
- Normalization (ImageNet stats)

### Custom Prediction Function
The `predict_video()` function:
- Extracts frames from any video
- Makes predictions on multiple frames
- Averages predictions for robust results
- Returns confidence scores

## 🚀 Future Improvements

- [ ] **Model Enhancements**
  - Try different architectures (ResNet, Xception, Vision Transformers)
  - Implement ensemble methods
  - Add attention mechanisms

- [ ] **Data Processing**
  - Multi-scale frame analysis
  - Face detection and cropping
  - Temporal sequence modeling with LSTMs/GRU

- [ ] **Deployment**
  - REST API development
  - Web interface
  - Mobile app integration
  - Real-time video stream analysis

- [ ] **Performance**
  - Model quantization
  - TensorRT optimization
  - Edge device deployment

- [ ] **Robustness**
  - Cross-validation
  - Adversarial training
  - Test on diverse datasets

## 📚 Project Structure

```
deepfake-detection/
├── project.py                      # Main script
├── README.md                       # Documentation
├── requirements.txt                # Dependencies
├── Celeb-real/                    # Real videos
├── Celeb-synthesis/               # Fake videos
├── YouTube-real/                  # Real videos
├── best_model.pth                 # Best model (generated)
├── deepfake_detector_final.pth    # Final model (generated)
├── training_history.png           # Training plots (generated)
├── confusion_matrix.png           # Confusion matrix (generated)
├── sample_predictions.png         # Predictions (generated)
└── model_config.json              # Config (generated)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Important Notes

- **Ethical Use**: This tool is for research and educational purposes. Always respect privacy and legal guidelines when collecting or analyzing video data.
- **Dataset Quality**: Model performance heavily depends on dataset quality and diversity.
- **GPU Recommended**: Training is significantly faster with CUDA-enabled GPU.
- **Memory Requirements**: Ensure adequate RAM/VRAM for batch processing.

## 📝 Requirements File

Create a `requirements.txt` with:

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=10.0.0
```

## 🎓 References

- EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- Deepfake Detection: Various academic papers on deepfake detection methods
- Transfer Learning: Fine-tuning pre-trained models for specific tasks



## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- EfficientNet authors for the architecture
- Dataset contributors
- Open-source community

---


## ⭐ Star This Repository

If you find this project helpful, please consider giving it a star!

---

**Disclaimer**: This is a research project. Deepfake detection is an evolving field, and no detector is 100% accurate. Always verify important content through multiple sources.
