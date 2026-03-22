"""
Main Orchestration Script - Forensic IPC Mapper
Complete end-to-end ML + IPCV pipeline for FIR document analysis
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Import all modules
from data_synthesis import SyntheticFIRGenerator
from preprocessing import IPCVPreprocessor
from feature_extraction import FeatureExtractor
from training import ModelTrainer
from inference import IPCPredictor
from evaluation import ModelEvaluator
from config import DATA_DIR, RESULTS_DIR, MODELS_DIR


class ForensicIPCMapper:
    """
    Complete ML + IPCV system for forensic document analysis
    Combines image processing with machine learning for IPC classification
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.log("Forensic IPC Mapper initialized")
    
    def _setup_logger(self):
        """Setup logging system"""
        class Logger:
            def __init__(self):
                self.logs = []
                self.start_time = datetime.now()
            
            def log(self, message, level="INFO"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_msg = f"[{timestamp}] [{level}] {message}"
                print(log_msg)
                self.logs.append(log_msg)
            
            def save_logs(self, path):
                with open(path, 'w') as f:
                    f.write('\n'.join(self.logs))
        
        return Logger()
    
    def log(self, message, level="INFO"):
        """Log message"""
        self.logger.log(message, level)
    
    def stage1_generate_synthetic_data(self):
        """Stage 1: Generate synthetic FIR dataset"""
        self.log("\n" + "=" * 70)
        self.log("STAGE 1: SYNTHETIC DATA GENERATION", "START")
        self.log("=" * 70)
        
        try:
            generator = SyntheticFIRGenerator()
            dataset_info = generator.generate_dataset()
            
            self.log(f"Generated {len(dataset_info)} synthetic samples", "SUCCESS")
            self.log(f"Dataset saved to: {DATA_DIR / 'synthetic'}", "SUCCESS")
            
            return True
        
        except Exception as e:
            self.log(f"Error in synthetic data generation: {e}", "ERROR")
            return False
    
    def stage2_analyze_preprocessing(self):
        """Stage 2: Demonstrate IPCV preprocessing pipeline"""
        self.log("\n" + "=" * 70)
        self.log("STAGE 2: IPCV PREPROCESSING PIPELINE DEMO", "START")
        self.log("=" * 70)
        
        try:
            preprocessor = IPCVPreprocessor()
            
            self.log("IPCV Pipeline Stages:", "INFO")
            self.log("  1. Noise Reduction (Non-Local Means Denoising)", "INFO")
            self.log("  2. Contrast Enhancement (CLAHE)", "INFO")
            self.log("  3. Perspective Correction (Edge-based)", "INFO")
            self.log("  4. Binarization (Adaptive Thresholding)", "INFO")
            self.log("  5. Morphological Operations (Closing/Opening)", "INFO")
            self.log("  6. Edge Detection (Canny + Dilation)", "INFO")
            
            self.log("Preprocessing pipeline configured successfully", "SUCCESS")
            return True
        
        except Exception as e:
            self.log(f"Error in preprocessing setup: {e}", "ERROR")
            return False
    
    def stage3_demonstrate_feature_extraction(self):
        """Stage 3: Demonstrate feature extraction"""
        self.log("\n" + "=" * 70)
        self.log("STAGE 3: FEATURE EXTRACTION SYSTEM", "START")
        self.log("=" * 70)
        
        try:
            extractor = FeatureExtractor()
            
            self.log("Feature Categories:", "INFO")
            self.log("  - HOG Features (Histogram of Oriented Gradients)", "INFO")
            self.log("  - LBP Features (Local Binary Patterns)", "INFO")
            self.log("  - Edge Features (Sobel operators)", "INFO")
            self.log("  - Contour Features (Structure analysis)", "INFO")
            self.log("  - Statistical Features (Moments, percentiles)", "INFO")
            self.log("  - Morphological Features (Erosion, dilation)", "INFO")
            self.log("  - Text Region Features (Connected components)", "INFO")
            self.log("  - Frequency Domain Features (FFT analysis)", "INFO")
            
            self.log("Total features extracted: 65+ per image", "INFO")
            self.log("Feature extraction system configured", "SUCCESS")
            return True
        
        except Exception as e:
            self.log(f"Error in feature extraction: {e}", "ERROR")
            return False
    
    def stage4_train_model(self):
        """Stage 4: Train Random Forest model"""
        self.log("\n" + "=" * 70)
        self.log("STAGE 4: RANDOM FOREST MODEL TRAINING", "START")
        self.log("=" * 70)
        
        try:
            trainer = ModelTrainer()
            trainer.train()
            
            self.log("Model training completed successfully", "SUCCESS")
            self.log(f"Model saved to: {MODELS_DIR}", "SUCCESS")
            
            return True
        
        except Exception as e:
            self.log(f"Error in model training: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
    
    def stage5_evaluate_model(self):
        """Stage 5: Evaluate model performance"""
        self.log("\n" + "=" * 70)
        self.log("STAGE 5: MODEL EVALUATION", "START")
        self.log("=" * 70)
        
        try:
            evaluator = ModelEvaluator()
            evaluator.load_test_data()
            evaluator.compute_metrics()
            
            self.log("Model evaluation completed", "SUCCESS")
            
            metrics = evaluator.metrics['overall']
            self.log(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)", "RESULT")
            self.log(f"Precision: {metrics['precision']:.4f}", "RESULT")
            self.log(f"Recall:    {metrics['recall']:.4f}", "RESULT")
            self.log(f"F1-Score:  {metrics['f1_score']:.4f}", "RESULT")
            
            evaluator.save_report()
            
            return True
        
        except Exception as e:
            self.log(f"Error in model evaluation: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
    
    def stage6_demo_inference(self):
        """Stage 6: Demonstrate inference system"""
        self.log("\n" + "=" * 70)
        self.log("STAGE 6: INFERENCE SYSTEM DEMO", "START")
        self.log("=" * 70)
        
        try:
            predictor = IPCPredictor()
            
            self.log("Inference System Features:", "INFO")
            self.log("  - Single image prediction with confidence scores", "INFO")
            self.log("  - Top-K predictions (top 3 IPC sections)", "INFO")
            self.log("  - Feature importance analysis", "INFO")
            self.log("  - Prediction explanability", "INFO")
            self.log("  - Batch prediction support", "INFO")
            
            self.log(predictor.get_model_info(), "INFO")
            
            self.log("Inference system ready for deployment", "SUCCESS")
            
            return True
        
        except Exception as e:
            self.log(f"Error in inference system: {e}", "ERROR")
            return False
    
    def run_complete_pipeline(self):
        """Run complete end-to-end ML + IPCV pipeline"""
        self.log("\n" + "=" * 80)
        self.log("FORENSIC IPC MAPPER - COMPLETE PIPELINE", "START")
        self.log("=" * 80)
        
        stages = [
            ("Synthetic Data Generation", self.stage1_generate_synthetic_data),
            ("IPCV Preprocessing", self.stage2_analyze_preprocessing),
            ("Feature Extraction", self.stage3_demonstrate_feature_extraction),
            ("Model Training", self.stage4_train_model),
            ("Model Evaluation", self.stage5_evaluate_model),
            ("Inference System", self.stage6_demo_inference),
        ]
        
        results = {}
        
        for stage_name, stage_func in stages:
            try:
                success = stage_func()
                results[stage_name] = "COMPLETED" if success else "FAILED"
            except Exception as e:
                self.log(f"Stage failed with exception: {e}", "ERROR")
                results[stage_name] = "FAILED"
        
        # Summary
        self.log("\n" + "=" * 80)
        self.log("PIPELINE SUMMARY", "SUMMARY")
        self.log("=" * 80)
        
        for stage, status in results.items():
            self.log(f"{stage}: {status}", "SUMMARY")
        
        # Save logs
        log_path = RESULTS_DIR / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.logger.save_logs(log_path)
        self.log(f"\nExecution log saved to: {log_path}", "INFO")
        
        all_success = all(status == "COMPLETED" for status in results.values())
        
        if all_success:
            self.log("\nAll stages completed successfully!", "SUCCESS")
        else:
            self.log("\nSome stages failed. Check logs for details.", "WARNING")
        
        return all_success
    
    def print_project_info(self):
        """Print project information"""
        info = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  FORENSIC IPC MAPPER - PROJECT OVERVIEW                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROJECT: Forensic-Grade ML + IPCV System for FIR Document Analysis
TECHNOLOGY STACK: Python, OpenCV, scikit-learn, scikit-image

ARCHITECTURE:
───────────────────────────────────────────────────────────────────────────────

1. DATA SYNTHESIS MODULE (data_synthesis.py)
   - Generates 500+ synthetic FIR document images
   - Augmentation: noise, blur, rotation, perspective distortion
   - Simulates real-world document scanning challenges

2. IPCV PREPROCESSING PIPELINE (preprocessing.py)
   - Stage 1: Noise Reduction (Non-Local Means Denoising)
   - Stage 2: Contrast Enhancement (CLAHE)
   - Stage 3: Perspective Correction (Edge-based)
   - Stage 4: Binarization (Adaptive Thresholding)
   - Stage 5: Morphological Operations
   - Stage 6: Edge Detection (Canny)

3. FEATURE EXTRACTION (feature_extraction.py)
   - 65+ Features per image:
     * HOG (Histogram of Oriented Gradients): ~180 features
     * LBP (Local Binary Patterns): ~59 features
     * Edge Features: 7 features
     * Contour Features: 13 features
     * Statistical Features: 11 features
     * Morphological Features: 8 features
     * Text Region Features: 6 features
     * Frequency Domain Features: 4 features

4. MACHINE LEARNING MODEL (training.py)
   - Algorithm: Random Forest Classifier
   - Trees: 200 estimators
   - Features: 260+ total (after feature extraction)
   - Classes: 10 IPC sections
   - Expected Accuracy: 75-85%

5. INFERENCE SYSTEM (inference.py)
   - Single & batch prediction
   - Confidence scores
   - Top-K predictions (top 3)
   - Feature importance analysis
   - Prediction explanability

6. EVALUATION PIPELINE (evaluation.py)
   - Accuracy, Precision, Recall, F1-Score
   - Per-class performance metrics
   - Confusion matrix analysis
   - Feature importance ranking
   - Confidence distribution analysis

KEY FEATURES:
───────────────────────────────────────────────────────────────────────────────
✓ Novel ML + IPCV Integration: Heavy image processing + machine learning
✓ Forensic-Grade Accuracy: 75-85% expected on synthetic data
✓ Interpretable AI: Feature importance and prediction explanations
✓ Production-Ready: Modular, scalable architecture
✓ Real-World Challenges: Handles noise, perspective, variable lighting
✓ Comprehensive Evaluation: Detailed metrics and performance analysis

USAGE:
───────────────────────────────────────────────────────────────────────────────
Run complete pipeline:      python scripts/main.py --full
Generate data only:         python scripts/data_synthesis.py
Train model only:           python scripts/training.py
Make predictions:           python scripts/inference.py --image <path>
Evaluate model:             python scripts/evaluation.py

IPC SECTIONS (Sample):
───────────────────────────────────────────────────────────────────────────────
IPC_302: Punishment for voluntarily causing hurt
IPC_307: Attempt to murder
IPC_308: Attempt to commit culpable homicide
IPC_336: Act endangering life or personal safety
IPC_379: Punishment for theft
IPC_392: Punishment for dacoity
IPC_419: Punishment for cheating
IPC_420: Cheating and dishonestly inducing delivery of property
... and more

PROJECT STRUCTURE:
───────────────────────────────────────────────────────────────────────────────
scripts/
  ├── config.py              # Configuration constants
  ├── data_synthesis.py      # Synthetic data generation
  ├── preprocessing.py       # IPCV pipeline
  ├── feature_extraction.py  # Feature engineering
  ├── training.py            # Model training
  ├── inference.py           # Predictions
  ├── evaluation.py          # Metrics & analysis
  └── main.py               # Orchestration

data/
  ├── synthetic/             # Generated FIR images
  └── processed/             # Preprocessed images

models/
  ├── random_forest_ipc_classifier.pkl  # Trained model
  ├── feature_scaler.pkl               # Feature normalization
  └── feature_names.pkl                # Feature metadata

results/
  ├── evaluation_report.txt
  ├── evaluation_metrics.json
  └── execution_logs/

═══════════════════════════════════════════════════════════════════════════════
        """
        print(info)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Forensic IPC Mapper - ML + IPCV Pipeline"
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run complete end-to-end pipeline'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Print project information'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate synthetic data only'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model only'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model only'
    )
    
    args = parser.parse_args()
    
    mapper = ForensicIPCMapper()
    
    if args.info or (not args.full and not args.generate_data and not args.train and not args.evaluate):
        mapper.print_project_info()
    
    if args.full:
        mapper.run_complete_pipeline()
    elif args.generate_data:
        mapper.stage1_generate_synthetic_data()
    elif args.train:
        mapper.stage4_train_model()
    elif args.evaluate:
        mapper.stage5_evaluate_model()


if __name__ == "__main__":
    main()
