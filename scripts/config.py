"""
Configuration constants for Forensic IPC Mapper
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SYNTHETIC_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Synthetic data generation parameters
SYNTHETIC_SAMPLES_PER_IPC = 50  # 50 samples per IPC section = ~500 total samples
SYNTHETIC_IMAGE_SIZE = (400, 600)  # (height, width) - typical document size
NOISE_LEVELS = [0.02, 0.05, 0.08]  # Gaussian noise variance
BLUR_KERNELS = [3, 5, 7]  # Kernel sizes for blur augmentation
ROTATION_ANGLES = [-5, -2, 0, 2, 5]  # Rotation angles in degrees
PERSPECTIVE_DISTORTION_RANGE = 0.1  # Max distortion factor

# IPC Sections database (mapping major categories)
IPC_SECTIONS = {
    "IPC_302": "Punishment for voluntarily causing hurt",
    "IPC_307": "Attempt to murder",
    "IPC_308": "Attempt to commit culpable homicide",
    "IPC_336": "Act endangering life or personal safety",
    "IPC_337": "Causing hurt by act endangering life",
    "IPC_379": "Punishment for theft",
    "IPC_380": "Theft in dwelling house etc",
    "IPC_392": "Punishment for dacoity",
    "IPC_419": "Punishment for cheating",
    "IPC_420": "Cheating and dishonestly inducing delivery of property",
}

# Feature extraction parameters
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = 8
HOG_CELLS_PER_BLOCK = 2
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

# Model training parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1  # Use all cores

# Inference parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum probability to consider a prediction
TOP_K_PREDICTIONS = 3  # Return top K predictions

# Model file paths
MODEL_PATH = MODELS_DIR / "random_forest_ipc_classifier.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# OCR parameters (for text extraction)
OCR_LANGUAGE = "eng"  # Tesseract language
OCR_CONFIG = "--psm 6"  # Page segmentation mode
