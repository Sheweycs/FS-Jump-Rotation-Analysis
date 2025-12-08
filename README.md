# FS-Jump-Rotation-Analysis

**Jump in Figure Skating: AI-Driven Rotation Analysis**

A deep learning-based system for automatically detecting under-rotations in figure skating jumps using motion data extracted from video.

## ⛸️ Overview

In figure skating competitions, judging accuracy and fairness are essential but often limited by human perception. One of the most common judging errors arises from under-rotations—cases where a skater's jump rotation is slightly insufficient upon landing. Detecting such under-rotations by eye is difficult, particularly in fast multi-rotation jumps and when camera angles change.

This project aims to build a deep learning-based system that can automatically determine whether a jump is under-rotated using motion data extracted from video. The system provides:

1. A transparent, physics-consistent method of evaluating jump rotation
2. Reduced bias introduced by human judgment
3. Potential use as a tool to assist technical specialists in real-time scoring or post-event video review

## 🌼 Project Structure

```
FS-Jump-Rotation-Analysis/
├── MAIN_FS_Rotation_Analysis.ipynb   # Main notebook with the whole workflow
├── FS_LSTM_JUMP.ipynb                # LSTM model training record
│
├── # Core Model
├── tcn_model.py                      # TCN model architecture (TCNBlock, TinyTCN)
├── Building_LSTM_Model.py            # LSTM-based model for jumps/non-jumps classification
├── training.py                       # Training and evaluation functions
├── hyperparameter_tuning.py          # Hyperparameter optimization using Optuna
│
├── # Data Processing
├── coords_extraction.py              # Extract skeleton coordinates from videos using MediaPipe
├── skeleton_video_extraction.py      # Generate skeleton overlay videos from original videos
├── Jump_candidates_segmentation.py   # Physics-based jump candidate segmentation
├── data_augmentation.py              # Video augmentation (flip, rotate) for dataset balancing
│
├── # Visualization
├── result_visualization.py           # Overlay prediction results on skeleton videos
├── Result_Compare_Visualization.py   # Analyze predictions and visualize decision space (t-SNE)
│
├── # Config & Docs
├── environment.yml                   # Conda environment configuration
├── system workflow.png               # System architecture diagram
└── README.md                         # This file
```

## ✨ Features

### 1. **Pose Extraction**
- Uses MediaPipe Pose to extract 33 anatomical landmarks per frame
- Processes videos at ~15 FPS for efficient pose detection
- Normalizes and preprocesses skeleton data

### 2. **Temporal Convolutional Network (TCN)**
- **TCNBlock**: Residual blocks with dilated convolutions for temporal pattern recognition
- **TinyTCN**: Lightweight architecture with:
  - Three TCN blocks with increasing dilation rates (1, 2, 4)
  - Global Average Pooling and Global Max Pooling
  - Classification head for rotation sufficiency detection

### 3. **Training Pipeline**
- Modular training and evaluation functions
- Support for class-weighted loss functions
- Gradient clipping for stable training
- Comprehensive metrics: accuracy, F1-score, recall, ROC-AUC

### 4. **Hyperparameter Optimization**
- Bayesian optimization using Optuna
- Supports both F1-score and recall-based optimization
- Efficient search space exploration

### 5. **Jump Segmentation**
- Physics-based heuristics for jump candidate detection
- Body tightness and angular velocity analysis
- Automatic clip extraction from full routines

## 🔧 Installation

### Option 1: Using Conda (Recommended)

Create and activate the conda environment:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate fs-jump-rotation-analysis
```

### Option 2: Using pip

Install dependencies using pip:

```bash
# Core dependencies
pip install torch torchvision
pip install numpy scikit-learn
pip install optuna
pip install opencv-python mediapipe
pip install pandas matplotlib

# Optional (for notebook)
pip install jupyter notebook
pip install rich torchmetrics
```

### Requirements

The project requires:
- **Python**: 3.10 or higher
- **PyTorch**: 2.0.0 or higher
- **NumPy**: 1.21.0 or higher
- **scikit-learn**: 1.0.0 or higher
- **Optuna**: 3.0.0 or higher (for hyperparameter tuning)
- **MediaPipe**: 0.10.0 or higher (for pose estimation)
- **OpenCV**: 4.5.0 or higher (for video processing)
- **Pandas**: 1.3.0 or higher
- **Matplotlib**: 3.5.0 or higher

Optional dependencies (for running notebooks):
- **Jupyter**: 1.0.0 or higher
- **Notebook**: 6.5.0 or higher
- **Rich**: 13.0.0 or higher (for enhanced terminal output)
- **TorchMetrics**: 0.11.0 or higher (for additional metrics)

## 📝 Dataset

The project uses the "Figure Skating Under-rotations and Flutz/Lip" dataset from [Kaggle](https://www.kaggle.com/datasets/sarazany/figure-skating-underrotations-and-flutzlip/data).

The dataset consists of short video clips from international figure skating competitions, categorized by:
- **Full rotation**: Complete jump rotations
- **Under-rotation**: Insufficient rotations (our focus)
- **Questionable (q)**: Borderline cases
- **Downgraded (down)**: Severely under-rotated

## 🔬 Methodology

1. **Pose Extraction**: MediaPipe Pose extracts 33 keypoints per frame
2. **Preprocessing**: Normalization and temporal alignment
3. **Model Architecture**: TCN with dilated convolutions for temporal pattern recognition
4. **Training**: Cross-entropy loss with class weights, AdamW optimizer
5. **Evaluation**: Multiple metrics including accuracy, F1-score, recall, and ROC-AUC

## 📄 License

This project is part of an academic course project (MIE1517).

## 🙏 Acknowledgments

- MediaPipe team for pose estimation framework
- Optuna for hyperparameter optimization
- Kaggle dataset contributors

---

For detailed implementation and experimental results, please refer to `MAIN_FS_Rotation_Analysis.ipynb`.
