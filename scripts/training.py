"""
Random Forest Model Training Pipeline
Trains classifier on synthetic FIR dataset with feature scaling
"""

import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pickle
import warnings

from config import (
    SYNTHETIC_DATA_DIR, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    TRAIN_TEST_SPLIT, RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF, RF_N_JOBS, IPC_SECTIONS
)
from feature_extraction import FeatureExtractor
from preprocessing import IPCVPreprocessor

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train Random Forest classifier for IPC section prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_metrics = {}
    
    def load_features_from_dataset(self):
        """Load preprocessed images and extract features"""
        print("Loading synthetic dataset and extracting features...")
        
        synthetic_dir = SYNTHETIC_DATA_DIR
        
        # Check if dataset exists
        image_files = list(synthetic_dir.glob("*.png"))
        
        if not image_files:
            raise FileNotFoundError(
                f"No synthetic data found in {synthetic_dir}. "
                "Run data_synthesis.py first."
            )
        
        X = []
        y = []
        failed_count = 0
        
        extractor = FeatureExtractor()
        
        for idx, img_file in enumerate(image_files):
            try:
                # Extract IPC code from filename (format: IPC_XXX_YYY.png)
                ipc_code = img_file.stem.split('_')[0:2]
                ipc_code = '_'.join(ipc_code)
                
                # Extract features
                features, feature_names = extractor.extract_all_features(str(img_file))
                
                X.append(features)
                y.append(ipc_code)
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(image_files)} images")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                failed_count += 1
        
        if not X:
            raise RuntimeError("No features could be extracted from dataset")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"Failed samples: {failed_count}")
        print(f"Feature dimensions: {X.shape}")
        print(f"Unique IPC sections: {len(np.unique(y))}")
        
        # Store feature names
        self.feature_names = feature_names
        
        return X, y
    
    def create_label_encoding(self, y):
        """Create mapping between IPC codes and integer labels"""
        unique_labels = np.unique(y)
        
        for idx, label in enumerate(unique_labels):
            self.label_encoder[label] = idx
            self.label_decoder[idx] = label
        
        # Convert labels to integers
        y_encoded = np.array([self.label_encoder[label] for label in y])
        
        print(f"Label encoding created:")
        for label, idx in sorted(self.label_encoder.items(), key=lambda x: x[1]):
            print(f"  {label} -> {idx}")
        
        return y_encoded
    
    def train(self):
        """
        Complete training pipeline:
        1. Load features from synthetic dataset
        2. Split into train/test
        3. Scale features
        4. Train Random Forest
        5. Evaluate and save model
        """
        print("\n" + "=" * 60)
        print("RANDOM FOREST TRAINING PIPELINE")
        print("=" * 60 + "\n")
        
        # Step 1: Load features
        X, y = self.load_features_from_dataset()
        
        # Step 2: Encode labels
        y_encoded = self.create_label_encoding(y)
        
        # Step 3: Train/test split
        print("\nSplitting dataset...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=1 - TRAIN_TEST_SPLIT,
            random_state=RANDOM_STATE,
            stratify=y_encoded
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Step 4: Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Step 5: Train Random Forest
        print("\nTraining Random Forest...")
        print(f"Parameters:")
        print(f"  n_estimators: {RF_N_ESTIMATORS}")
        print(f"  max_depth: {RF_MAX_DEPTH}")
        print(f"  min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
        print(f"  min_samples_leaf: {RF_MIN_SAMPLES_LEAF}")
        
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS,
            verbose=1
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("\nModel training completed!")
        
        # Step 6: Evaluate
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        self.evaluate()
        
        # Step 7: Save model
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        self.save_model()
    
    def evaluate(self):
        """Evaluate model on train and test sets"""
        # Training metrics
        y_train_pred = self.model.predict(self.X_train_scaled)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # Test metrics
        y_test_pred = self.model.predict(self.X_test_scaled)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed metrics
        print("\n" + "-" * 60)
        print("DETAILED TEST SET METRICS")
        print("-" * 60)
        
        precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        
        # Classification report
        print("\n" + "-" * 60)
        print("CLASSIFICATION REPORT")
        print("-" * 60)
        
        # Create label names
        label_names = [self.label_decoder[i] for i in range(len(self.label_decoder))]
        
        report = classification_report(
            self.y_test, y_test_pred,
            target_names=label_names,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(cm)
        
        # Feature importance
        print("\n" + "-" * 60)
        print("TOP 15 MOST IMPORTANT FEATURES")
        print("-" * 60)
        
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-15:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            feature_name = self.feature_names[idx]
            importance = feature_importance[idx]
            print(f"{rank:2d}. {feature_name:30s} {importance:.6f}")
        
        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_features': len(self.feature_names),
            'n_training_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'n_classes': len(self.label_decoder),
        }
    
    def save_model(self):
        """Save trained model, scaler, and metadata"""
        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to: {MODEL_PATH}")
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {SCALER_PATH}")
        
        # Save feature names
        with open(FEATURE_NAMES_PATH, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Feature names saved to: {FEATURE_NAMES_PATH}")
        
        # Save metadata
        metadata = {
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'training_metrics': self.training_metrics,
            'ipc_sections': IPC_SECTIONS,
            'model_params': {
                'n_estimators': RF_N_ESTIMATORS,
                'max_depth': RF_MAX_DEPTH,
                'min_samples_split': RF_MIN_SAMPLES_SPLIT,
                'min_samples_leaf': RF_MIN_SAMPLES_LEAF,
                'random_state': RANDOM_STATE,
            }
        }
        
        metadata_path = Path(MODEL_PATH).parent / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        print("\nAll model files saved successfully!")
    
    def load_model(self):
        """Load trained model and associated files"""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load metadata
        metadata_path = Path(MODEL_PATH).parent / "model_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.label_encoder = metadata['label_encoder']
        self.label_decoder = {int(k): v for k, v in metadata['label_decoder'].items()}
        
        print("Model loaded successfully!")


def main():
    """Train the IPC classifier model"""
    trainer = ModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
