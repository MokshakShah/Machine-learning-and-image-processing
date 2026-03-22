"""
Evaluation and Metrics Pipeline
Comprehensive performance analysis and reporting
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
import pickle

from config import (
    MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    SYNTHETIC_DATA_DIR, RESULTS_DIR, IPC_SECTIONS
)
from feature_extraction import FeatureExtractor
from training import ModelTrainer


class ModelEvaluator:
    """Comprehensive evaluation of IPC classifier model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.metrics = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        trainer = ModelTrainer()
        trainer.load_model()
        
        self.model = trainer.model
        self.scaler = trainer.scaler
        self.feature_names = trainer.feature_names
        self.label_encoder = trainer.label_encoder
        self.label_decoder = trainer.label_decoder
    
    def load_test_data(self):
        """Load test dataset and features"""
        print("Loading test dataset...")
        
        synthetic_dir = SYNTHETIC_DATA_DIR
        image_files = list(synthetic_dir.glob("*.png"))
        
        X = []
        y = []
        
        extractor = FeatureExtractor()
        
        for idx, img_file in enumerate(image_files):
            try:
                ipc_code = '_'.join(img_file.stem.split('_')[0:2])
                features, _ = extractor.extract_all_features(str(img_file))
                
                X.append(features)
                y.append(ipc_code)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(image_files)} images")
            
            except Exception as e:
                print(f"Error: {e}")
        
        X = np.array(X)
        y_encoded = np.array([self.label_encoder[label] for label in y])
        
        self.X_test = self.scaler.transform(X)
        self.y_test = y_encoded
        
        print(f"Test data loaded: {len(X)} samples")
    
    def compute_metrics(self):
        """Compute comprehensive metrics"""
        print("\nComputing predictions...")
        
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)
        
        print("Computing metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(self.y_test, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_test, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_test, self.y_pred, average=None, zero_division=0)
        
        self.metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
            },
            'per_class': {}
        }
        
        # Per-class breakdown
        for class_idx in range(len(self.label_decoder)):
            ipc_code = self.label_decoder[class_idx]
            self.metrics['per_class'][ipc_code] = {
                'precision': float(precision_per_class[class_idx]),
                'recall': float(recall_per_class[class_idx]),
                'f1_score': float(f1_per_class[class_idx]),
            }
        
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = "FORENSIC IPC CLASSIFIER - EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Model configuration
        report += "MODEL CONFIGURATION\n"
        report += "-" * 70 + "\n"
        report += f"Model Type: Random Forest Classifier\n"
        report += f"Number of Trees: {self.model.n_estimators}\n"
        report += f"Max Depth: {self.model.max_depth}\n"
        report += f"Min Samples Split: {self.model.min_samples_split}\n"
        report += f"Min Samples Leaf: {self.model.min_samples_leaf}\n\n"
        
        # Dataset information
        report += "DATASET INFORMATION\n"
        report += "-" * 70 + "\n"
        report += f"Test Samples: {len(self.y_test)}\n"
        report += f"Number of Classes: {len(self.label_decoder)}\n"
        report += f"Number of Features: {len(self.feature_names)}\n\n"
        
        # Overall metrics
        report += "OVERALL PERFORMANCE METRICS\n"
        report += "-" * 70 + "\n"
        metrics = self.metrics['overall']
        report += f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n"
        report += f"Precision: {metrics['precision']:.4f}\n"
        report += f"Recall:    {metrics['recall']:.4f}\n"
        report += f"F1-Score:  {metrics['f1_score']:.4f}\n\n"
        
        # Per-class metrics
        report += "PER-CLASS PERFORMANCE\n"
        report += "-" * 70 + "\n"
        report += f"{'IPC Code':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
        report += "-" * 70 + "\n"
        
        for ipc_code, metrics in sorted(self.metrics['per_class'].items()):
            report += f"{ipc_code:<15} "
            report += f"{metrics['precision']:<12.4f} "
            report += f"{metrics['recall']:<12.4f} "
            report += f"{metrics['f1_score']:<12.4f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        report += "FEATURE IMPORTANCE (Top 20)\n"
        report += "=" * 70 + "\n"
        
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-20:][::-1]
        
        cumulative_importance = 0
        for rank, idx in enumerate(top_indices, 1):
            importance = feature_importance[idx]
            cumulative_importance += importance
            feature_name = self.feature_names[idx]
            
            report += f"{rank:2d}. {feature_name:<30} {importance:.6f} "
            report += f"(Cumulative: {cumulative_importance:.4f})\n"
        
        report += "\n" + "=" * 70 + "\n"
        report += "CONFUSION MATRIX ANALYSIS\n"
        report += "=" * 70 + "\n"
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        report += f"\nConfusion Matrix Shape: {cm.shape}\n"
        report += f"Total Correct: {np.trace(cm)}/{len(self.y_test)}\n"
        report += f"Misclassifications: {len(self.y_test) - np.trace(cm)}\n"
        
        return report
    
    def analyze_confidence(self):
        """Analyze prediction confidence distribution"""
        max_proba = np.max(self.y_proba, axis=1)
        correct = (self.y_pred == self.y_test)
        
        correct_confidence = max_proba[correct]
        incorrect_confidence = max_proba[~correct]
        
        analysis = "CONFIDENCE ANALYSIS\n"
        analysis += "=" * 60 + "\n\n"
        
        analysis += "Overall Confidence Statistics:\n"
        analysis += f"  Mean: {np.mean(max_proba):.4f}\n"
        analysis += f"  Std:  {np.std(max_proba):.4f}\n"
        analysis += f"  Min:  {np.min(max_proba):.4f}\n"
        analysis += f"  Max:  {np.max(max_proba):.4f}\n\n"
        
        analysis += "Correct Predictions:\n"
        analysis += f"  Count: {len(correct_confidence)}\n"
        analysis += f"  Mean Confidence: {np.mean(correct_confidence):.4f}\n"
        analysis += f"  Std:  {np.std(correct_confidence):.4f}\n\n"
        
        if len(incorrect_confidence) > 0:
            analysis += "Incorrect Predictions:\n"
            analysis += f"  Count: {len(incorrect_confidence)}\n"
            analysis += f"  Mean Confidence: {np.mean(incorrect_confidence):.4f}\n"
            analysis += f"  Std:  {np.std(incorrect_confidence):.4f}\n\n"
            
            analysis += f"Confidence Difference: {np.mean(correct_confidence) - np.mean(incorrect_confidence):.4f}\n"
        
        return analysis
    
    def save_report(self):
        """Save evaluation report to file"""
        report = self.generate_report()
        report += "\n\n"
        report += self.analyze_confidence()
        
        report_path = RESULTS_DIR / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
        
        # Also save metrics as JSON
        metrics_path = RESULTS_DIR / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to: {metrics_path}")
    
    def generate_summary(self):
        """Generate executive summary"""
        metrics = self.metrics['overall']
        
        summary = "EXECUTIVE SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Accuracy: {metrics['accuracy']*100:.2f}%\n"
        summary += f"F1-Score: {metrics['f1_score']:.4f}\n\n"
        
        summary += "Model Status: "
        if metrics['accuracy'] > 0.85:
            summary += "Excellent - Ready for production\n"
        elif metrics['accuracy'] > 0.75:
            summary += "Good - Suitable for most applications\n"
        elif metrics['accuracy'] > 0.65:
            summary += "Fair - Needs improvement\n"
        else:
            summary += "Poor - Requires retraining\n"
        
        return summary


def evaluate_model():
    """Run complete evaluation pipeline"""
    evaluator = ModelEvaluator()
    
    try:
        evaluator.load_test_data()
        evaluator.compute_metrics()
        
        print("\n" + evaluator.generate_summary())
        
        print("\n" + evaluator.generate_report())
        print("\n" + evaluator.analyze_confidence())
        
        evaluator.save_report()
        
        print("\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main evaluation entry point"""
    evaluate_model()


if __name__ == "__main__":
    main()
