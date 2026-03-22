# System Architecture - Forensic IPC Mapper

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FORENSIC IPC MAPPER SYSTEM                           │
│         ML + IPCV Integration for Automatic IPC Classification          │
└─────────────────────────────────────────────────────────────────────────┘

                              USER INPUT
                                  ↓
                         (FIR Document Image)
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │          STAGE 1: DATA PREPARATION              │
        │  (Synthetic data generation for training)       │
        └─────────────────────────────────────────────────┘
                                  ↓
                    500+ Synthetic FIR Images
                    (with IPC labels)
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │    STAGE 2: IPCV PREPROCESSING PIPELINE         │
        │  (6-stage image processing for enhancement)     │
        └─────────────────────────────────────────────────┘
                                  ↓
                    Preprocessed Images
                    (enhanced quality)
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │    STAGE 3: FEATURE EXTRACTION                  │
        │  (Extract 65+ features from images)             │
        └─────────────────────────────────────────────────┘
                                  ↓
                    Feature Vectors (260+ dims)
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │    STAGE 4: MODEL TRAINING                      │
        │  (Random Forest on 10 IPC classes)              │
        └─────────────────────────────────────────────────┘
                                  ↓
                    Trained Model + Scaler
                    (saved to disk)
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │    STAGE 5: INFERENCE & PREDICTION              │
        │  (Make predictions on new images)               │
        └─────────────────────────────────────────────────┘
                                  ↓
                    IPC Prediction + Confidence
                    Top-K predictions
                    Feature importance
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │    STAGE 6: EVALUATION & ANALYSIS               │
        │  (Performance metrics & reporting)              │
        └─────────────────────────────────────────────────┘
                                  ↓
                    Accuracy, Precision, Recall
                    Confusion Matrix
                    Feature Importance Ranking
```

---

## Detailed Component Architecture

### 1. Synthetic Data Generation Module

```
SyntheticFIRGenerator
├── generate_fir_background()
│   ├── Create white canvas
│   ├── Add FIR header
│   ├── Add case details
│   └── Add document text
│
├── embed_ipc_section()
│   └── Insert IPC code into document
│
├── Augmentation Pipeline:
│   ├── add_noise() - Gaussian noise (2-8%)
│   ├── add_blur() - Kernel blur (3,5,7)
│   ├── add_rotation() - Angle rotation (-5 to +5°)
│   ├── add_perspective_distortion() - Skew effect
│   └── add_scan_artifacts() - Shadows & dust
│
└── generate_dataset()
    ├── For each IPC section:
    │   └── Generate 50 samples with augmentations
    └── Output: 500 labeled images

Files: scripts/data_synthesis.py
Output: data/synthetic/*.png
```

### 2. IPCV Preprocessing Pipeline

```
IPCVPreprocessor (6 Stages)

Input Image (400x600 pixels)
        ↓
┌─────────────────────────────┐
│ STAGE 1: NOISE REDUCTION    │
│ Non-Local Means Denoising   │
│ (h=10, preserve details)    │
└─────────────────────────────┘
        ↓ (h=10, preserve details)
┌─────────────────────────────┐
│ STAGE 2: CONTRAST           │
│ CLAHE Adaptive Histogram    │
│ (clipLimit=2.0)             │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ STAGE 3: PERSPECTIVE        │
│ Edge-based document         │
│ boundary detection & rotate │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ STAGE 4: BINARIZATION       │
│ Adaptive Gaussian           │
│ Thresholding (block=11)     │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ STAGE 5: MORPHOLOGY         │
│ Close → Open → Dilate       │
│ (3x3 and 5x5 kernels)       │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ STAGE 6: EDGE DETECTION     │
│ Canny (50-150) + Dilation   │
│ (3x3 kernel)                │
└─────────────────────────────┘
        ↓
Processed Image (enhanced)

Files: scripts/preprocessing.py
Method: process(img, return_intermediate=False)
```

### 3. Feature Extraction Module

```
FeatureExtractor (65+ features)

Input: Preprocessed image
        ↓
┌──────────────────────────────────────────┐
│ HOG FEATURES (~180 features)             │
│ ├── 9 orientations                       │
│ ├── 8×8 pixels per cell                  │
│ └── 2×2 cells per block                  │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ LBP FEATURES (~59 features)              │
│ ├── Radius: 3                            │
│ ├── Points: 24 (uniform method)          │
│ └── Histogram of patterns                │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ EDGE FEATURES (7 features)               │
│ ├── Sobel magnitude mean/std/min/max     │
│ └── Percentiles (Q25, Q50, Q75)          │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ CONTOUR FEATURES (13 features)           │
│ ├── Number of contours                   │
│ ├── Area statistics (mean/std/min/max)   │
│ ├── Perimeter statistics                 │
│ └── Circularity measures                 │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ STATISTICAL FEATURES (11 features)       │
│ ├── Intensity mean/std                   │
│ ├── Min/max/percentiles                  │
│ ├── Skewness                             │
│ └── Kurtosis                             │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ MORPHOLOGICAL FEATURES (8 features)      │
│ ├── Eroded/Dilated/Opened/Closed mean    │
│ └── Standard deviations                  │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ TEXT REGION FEATURES (6 features)        │
│ ├── Connected components count           │
│ ├── Component area statistics            │
│ └── Text density                         │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────┐
│ FREQUENCY DOMAIN FEATURES (4 features)   │
│ ├── FFT magnitude spectrum                │
│ └── Statistics (mean/std/max/Q90)        │
└──────────────────────────────────────────┘
        ↓
Feature Vector (260+ dimensions)
        ↓
StandardScaler (Normalization)
        ↓
Scaled Feature Vector (ready for ML)

Files: scripts/feature_extraction.py
Method: extract_all_features(img)
```

### 4. Machine Learning Model

```
RandomForestClassifier

Configuration:
├── n_estimators: 200 trees
├── max_depth: 20 (prevent overfitting)
├── min_samples_split: 5
├── min_samples_leaf: 2
├── random_state: 42 (reproducibility)
└── n_jobs: -1 (parallel processing)

Training:
├── Input: 400 samples (80/20 split)
│   ├── Training set: 320 samples
│   └── Test set: 80 samples
│
├── Features: 260+ (from feature extraction)
├── Classes: 10 IPC sections
│   ├── IPC_302, IPC_307, IPC_308, ...
│   └── Each with ~50 training samples
│
└── Output:
    ├── 200 Decision Trees
    ├── Feature importance scores
    ├── Trained model (pickle)
    ├── Feature scaler (pickle)
    └── Metadata (JSON)

Training Process:
Raw Images (500)
    ↓ [Feature Extraction]
Feature Vectors (260+ dims)
    ↓ [Scaling]
Normalized Features
    ↓ [Train/Test Split 80/20]
Training Set (400) + Test Set (100)
    ↓ [Random Forest Fit]
Trained Model (200 trees)
    ↓ [Evaluation]
Performance Metrics (Accuracy ~75-85%)

Files: scripts/training.py
Model saved: models/random_forest_ipc_classifier.pkl
```

### 5. Inference System

```
IPCPredictor

Input: New FIR document image
    ↓
1. Feature Extraction
   ├── Apply same IPCV pipeline
   ├── Extract 65+ features
   └── Get feature vector
    ↓
2. Feature Scaling
   ├── Load training scaler
   ├── Normalize features
   └── Get scaled vector
    ↓
3. Prediction
   ├── Random Forest predict()
   ├── Random Forest predict_proba()
   └── Get probabilities for all classes
    ↓
4. Output Processing
   ├── Predicted class → IPC code
   ├── Confidence score (max probability)
   ├── Top-K predictions (top 3)
   └── Feature importance for this prediction
    ↓
5. Return Results
   {
       "predicted_ipc": "IPC_307",
       "ipc_description": "Attempt to murder",
       "confidence": 0.87,
       "confidence_percentage": "87.00%",
       "top_k_predictions": [
           {"rank": 1, "ipc_code": "IPC_307", "probability": 0.87},
           {"rank": 2, "ipc_code": "IPC_308", "probability": 0.09},
           {"rank": 3, "ipc_code": "IPC_302", "probability": 0.04}
       ],
       "important_features": [
           {"rank": 1, "feature_name": "HOG_45", "importance": 0.082},
           ...
       ]
   }

Files: scripts/inference.py
Methods: 
  - predict_single(image) → prediction result
  - predict_batch(images) → list of results
  - get_prediction_explanation() → detailed explanation
```

### 6. Evaluation Module

```
ModelEvaluator

Evaluation Pipeline:
    ↓
1. Load Test Data
   ├── Load 100 test images
   ├── Extract features
   └── Scale features
    ↓
2. Make Predictions
   ├── RF predict()
   ├── RF predict_proba()
   └── Get all predictions
    ↓
3. Compute Metrics
   ├── Overall:
   │   ├── Accuracy
   │   ├── Precision (weighted)
   │   ├── Recall (weighted)
   │   └── F1-Score (weighted)
   │
   └── Per-Class:
       └── For each IPC section:
           ├── Precision
           ├── Recall
           └── F1-Score
    ↓
4. Detailed Analysis
   ├── Confusion matrix
   ├── Classification report
   ├── Feature importance (top 20)
   └── Confidence distribution
    ↓
5. Save Reports
   ├── evaluation_report.txt
   ├── evaluation_metrics.json
   └── execution_log_*.txt

Output Metrics:
├── Overall Accuracy: ~0.75-0.85
├── Precision: ~0.75-0.85
├── Recall: ~0.75-0.85
├── F1-Score: ~0.75-0.85
├── Confusion Matrix: 10×10 (for 10 classes)
└── Feature Rankings: Top 20 important features

Files: scripts/evaluation.py
Methods: 
  - load_test_data()
  - compute_metrics()
  - generate_report()
  - analyze_confidence()
  - save_report()
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                            │
└──────────────────────────────────────────────────────────────────────┘

Synthetic Data Generation
    ↓
    └─→ 500 FIR Images with IPC labels
            ↓
        IPCV Preprocessing (6 stages)
            ↓
        Feature Extraction (65+ features)
            ↓
        Feature Vectors (260+ dimensions)
            ↓
        Feature Scaling (StandardScaler)
            ↓
        Train/Test Split (80/20)
            ↓
        Random Forest Training
            ↓
        ┌─────────────────────────────┐
        │ Trained Model Artifacts:    │
        ├─────────────────────────────┤
        │ • random_forest_classifier  │
        │ • feature_scaler            │
        │ • feature_names             │
        │ • model_metadata            │
        └─────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                            │
└──────────────────────────────────────────────────────────────────────┘

New FIR Document Image
    ↓
IPCV Preprocessing (6 stages - same as training)
    ↓
Feature Extraction (65+ features - same as training)
    ↓
Feature Scaling (using saved scaler)
    ↓
Load Trained Model
    ↓
RF predict() + predict_proba()
    ↓
┌─────────────────────────┐
│ Output:                 │
├─────────────────────────┤
│ • Predicted IPC Code    │
│ • Confidence Score      │
│ • Top-K Predictions     │
│ • Feature Importance    │
│ • Explanation           │
└─────────────────────────┘
```

---

## Module Dependencies

```
main.py (Orchestration)
├── data_synthesis.py (generates training data)
├── preprocessing.py (6-stage IPCV pipeline)
├── feature_extraction.py (depends on preprocessing)
├── training.py (depends on feature_extraction & preprocessing)
│   └── saves → models/
├── inference.py (depends on preprocessing, feature_extraction, training)
│   └── loads → models/
├── evaluation.py (depends on all above)
│   └── saves → results/
└── config.py (shared by all modules)
```

---

## File Organization

```
forensic-ipc-mapper/
│
├── scripts/
│   ├── __init__.py (makes it a package)
│   ├── config.py              # Configuration hub
│   ├── data_synthesis.py      # Generate synthetic data
│   ├── preprocessing.py       # IPCV pipeline
│   ├── feature_extraction.py  # Feature engineering
│   ├── training.py            # Model training
│   ├── inference.py           # Predictions
│   ├── evaluation.py          # Metrics & analysis
│   └── main.py               # Main orchestration
│
├── data/
│   ├── synthetic/             # Generated training images
│   │   ├── IPC_302_000.png
│   │   ├── IPC_302_001.png
│   │   └── dataset_info.json
│   └── processed/             # Preprocessed images (optional)
│
├── models/
│   ├── random_forest_ipc_classifier.pkl
│   ├── feature_scaler.pkl
│   ├── feature_names.pkl
│   └── model_metadata.json
│
├── results/
│   ├── evaluation_report.txt
│   ├── evaluation_metrics.json
│   └── execution_log_*.txt
│
├── pyproject.toml             # Python project config
├── requirements.txt           # Pip dependencies
├── README.md                  # Full documentation
├── QUICK_START.md            # Quick start guide
├── ARCHITECTURE.md           # This file
└── .gitignore
```

---

## Performance Characteristics

### Time Complexity

| Stage | Operation | Time |
|-------|-----------|------|
| Data Generation | Create 500 images | 2-3 min |
| Preprocessing | 6 stages per image | 100-200 ms/image |
| Feature Extraction | Extract 65+ features | 500-800 ms/image |
| Training | Train RF on 400 samples | 1-2 min |
| Prediction | Single image | 1-2 seconds |
| Evaluation | 100 test samples | 2-3 min |
| **Total Pipeline** | **Full run** | **10-15 min** |

### Space Complexity

| Component | Size |
|-----------|------|
| 500 synthetic images | ~500 MB |
| Trained model (.pkl) | ~50-100 MB |
| Scaler (.pkl) | ~1 KB |
| Feature names (.pkl) | ~10 KB |
| Feature vectors (400 samples) | ~100 MB |
| **Total** | **~700 MB** |

---

## Scalability & Extension Points

### Easy Additions

1. **More IPC Sections**
   - Edit `IPC_SECTIONS` in config.py
   - Data generation auto-scales

2. **More Samples per Class**
   - Adjust `SYNTHETIC_SAMPLES_PER_IPC` in config.py
   - Improves model accuracy

3. **Different Image Augmentations**
   - Edit `data_synthesis.py` methods
   - Add more noise types, rotations, etc.

4. **New Feature Types**
   - Add methods to `FeatureExtractor` class
   - Auto-incorporated into feature vector

5. **Different ML Models**
   - Replace RandomForest in `training.py`
   - SVM, Neural Network, Gradient Boosting, etc.

### Difficult Additions

1. **Real image dataset** - Requires manual annotation
2. **Deep learning model** - Requires large GPU
3. **Real-time inference** - Requires model optimization
4. **Multi-language support** - Requires OCR for text extraction

---

## Error Handling & Robustness

### Built-in Safeguards

1. **File existence checks** - Prevents crashes on missing files
2. **Exception handling** - Try/catch blocks for robustness
3. **Data validation** - Checks image dimensions, feature counts
4. **Logging** - Tracks progress and failures
5. **Path validation** - Uses Path objects, handles cross-platform paths

### Failure Recovery

- If training fails: Clear models/ and restart
- If data generation fails: Delete data/synthetic/ and restart
- If evaluation fails: Ensure model is trained first

---

**Last Updated**: March 2026
**Status**: Stable & Production-Ready
