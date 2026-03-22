"""
Inference System with Confidence Scores
Predicts IPC sections from FIR document images with interpretability
"""

import numpy as np
import json
import cv2
import base64
from pathlib import Path
import pickle

from config import (
    MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    CONFIDENCE_THRESHOLD, TOP_K_PREDICTIONS, IPC_SECTIONS
)
from feature_extraction import FeatureExtractor
from training import ModelTrainer


class IPCPredictor:
    """
    Predict IPC sections from document images
    Provides confidence scores and top-K predictions
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoder = {}
        self.label_decoder = {}
        self.ipc_sections = IPC_SECTIONS
        self.extractor = FeatureExtractor()
        self.last_features = None
        self.last_prediction_details = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model and preprocessing artifacts"""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run training.py first to train the model."
            )
        
        try:
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
            
            print("[v0] Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict_single(self, img_path_or_array, return_details=True):
        """
        Predict IPC section from a single image
        
        Args:
            img_path_or_array: Path to image or numpy array
            return_details: If True, return confidence scores and top-K predictions
        
        Returns:
            Dict with prediction, confidence, and details
        """
        try:
            # Load raw image if path was provided for quality diagnostics.
            if isinstance(img_path_or_array, (str, Path)):
                raw_image = cv2.imread(str(img_path_or_array), cv2.IMREAD_GRAYSCALE)
                if raw_image is None:
                    raise ValueError(f"Could not load image: {img_path_or_array}")
            else:
                raw_image = img_path_or_array.copy()

            preprocessor = self.extractor.preprocessor
            best_refined_name, best_refined_image, best_refined_quality, variants = preprocessor.get_best_refined_image(raw_image)

            variant_probabilities = []
            variant_quality = {}
            best_features = None
            best_features_scaled = None
            best_variant_for_selected = "original"
            best_variant_confidence_for_selected = -1.0

            print("[v0] Extracting features from fused image variants...")
            for variant_name, variant_image in variants.items():
                variant_quality[variant_name] = preprocessor.analyze_image_quality(variant_image)
                features, _ = self.extractor.extract_all_features(variant_image)
                features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
                proba = self.model.predict_proba(features_scaled.reshape(1, -1))[0]

                variant_probabilities.append((variant_name, proba, features, features_scaled))

            # Fuse probabilities across all enhanced variants to reduce single-view noise.
            proba_stack = np.array([entry[1] for entry in variant_probabilities])
            proba = np.mean(proba_stack, axis=0)

            predicted_label = int(np.argmax(proba))
            confidence = float(proba[predicted_label])

            for variant_name, variant_proba, features, features_scaled in variant_probabilities:
                variant_conf = float(variant_proba[predicted_label])
                if variant_conf > best_variant_confidence_for_selected:
                    best_variant_confidence_for_selected = variant_conf
                    best_variant_for_selected = variant_name
                    best_features = features
                    best_features_scaled = features_scaled

            features = best_features
            features_scaled = best_features_scaled
            selected_variant = best_variant_for_selected

            self.last_features = features
            predicted_ipc = self.label_decoder[predicted_label]
            
            # Get IPC description
            ipc_description = self.ipc_sections.get(predicted_ipc, "Unknown IPC section")
            
            result = {
                'predicted_ipc': predicted_ipc,
                'ipc_description': ipc_description,
                'confidence': confidence,
                'confidence_percentage': f"{confidence * 100:.2f}%",
                'prediction_status': 'final_verdict',
                'trust_level': 'refined_pipeline',
                'is_reliable': True,
                'selected_image_variant': selected_variant,
                'best_refined_variant': best_refined_name,
                'image_quality': {
                    'original': variant_quality.get('original', {}),
                    'readability': variant_quality.get('readability', {}),
                    'stain_reduced': variant_quality.get('stain_reduced', {}),
                    'deblurred': variant_quality.get('deblurred', {}),
                    'deskewed': variant_quality.get('deskewed', {}),
                    'best_refined': best_refined_quality,
                    'used_variant': selected_variant,
                },
            }

            refined_for_preview = cv2.resize(best_refined_image, (700, 900), interpolation=cv2.INTER_CUBIC)
            ok, encoded = cv2.imencode('.jpg', refined_for_preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                result['refined_image_base64'] = base64.b64encode(encoded.tobytes()).decode('ascii')
                result['refined_image_mime'] = 'image/jpeg'
            
            if return_details:
                # Get top-K predictions
                top_k_indices = np.argsort(proba)[-TOP_K_PREDICTIONS:][::-1]
                
                top_k_predictions = []
                for rank, idx in enumerate(top_k_indices, 1):
                    ipc_code = self.label_decoder[idx]
                    prob = float(proba[idx])
                    
                    top_k_predictions.append({
                        'rank': rank,
                        'ipc_code': ipc_code,
                        'description': self.ipc_sections.get(ipc_code, "Unknown"),
                        'probability': prob,
                        'percentage': f"{prob * 100:.2f}%"
                    })
                
                result['top_k_predictions'] = top_k_predictions
                
                # Get feature importance for this prediction
                feature_importance = self.model.feature_importances_
                top_feature_indices = np.argsort(feature_importance)[-10:][::-1]
                
                important_features = []
                for rank, feat_idx in enumerate(top_feature_indices, 1):
                    important_features.append({
                        'rank': rank,
                        'feature_name': self.feature_names[feat_idx],
                        'importance': float(feature_importance[feat_idx]),
                        'importance_percentage': f"{feature_importance[feat_idx] * 100:.2f}%"
                    })
                
                result['important_features'] = important_features

                confidence_gap = 0.0
                if len(top_k_predictions) > 1:
                    confidence_gap = confidence - top_k_predictions[1]['probability']

                used_quality = variant_quality.get(selected_variant, {})
                quality_score = used_quality.get('quality_score', 0.0)

                reasons = []
                if used_quality.get('sharpness_score', 0.0) < 0.35:
                    reasons.append('image sharpness is low (blur or motion)')
                if used_quality.get('contrast_score', 0.0) < 0.35:
                    reasons.append('contrast is weak (text/background separation is poor)')
                if used_quality.get('brightness_score', 0.0) < 0.35:
                    reasons.append('lighting is uneven or over/under exposed')
                if confidence_gap < 0.12:
                    reasons.append('top IPC categories are close in probability')

                if not reasons and confidence < 0.35:
                    reasons.append('evidence in extracted features is weak for a single class')

                result['certainty_diagnostics'] = {
                    'quality_score': float(quality_score),
                    'confidence_gap': float(confidence_gap),
                    'reasons': reasons,
                }

                result['user_recommendation'] = (
                    'Verdict generated after multi-stage refinement (readability, stain suppression, deblur, deskew) '
                    'and fused model scoring across all refined variants.'
                )
            
            self.last_prediction_details = result
            return result
        
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def predict_batch(self, image_paths_or_arrays):
        """
        Predict IPC sections for multiple images
        
        Args:
            image_paths_or_arrays: List of image paths or numpy arrays
        
        Returns:
            List of prediction results
        """
        results = []
        
        for idx, img_input in enumerate(image_paths_or_arrays):
            print(f"Processing image {idx + 1}/{len(image_paths_or_arrays)}...")
            result = self.predict_single(img_input, return_details=False)
            results.append(result)
        
        return results
    
    def get_prediction_explanation(self):
        """
        Get detailed explanation of last prediction
        Shows feature contributions and model reasoning
        """
        if self.last_prediction_details is None:
            return "No prediction made yet"
        
        explanation = "PREDICTION EXPLANATION\n"
        explanation += "=" * 60 + "\n\n"
        
        details = self.last_prediction_details
        
        explanation += f"Predicted IPC Section: {details['predicted_ipc']}\n"
        explanation += f"Description: {details['ipc_description']}\n"
        explanation += f"Confidence: {details['confidence_percentage']}\n\n"
        
        if 'top_k_predictions' in details:
            explanation += "Top 3 Predictions:\n"
            explanation += "-" * 60 + "\n"
            for pred in details['top_k_predictions']:
                explanation += f"{pred['rank']}. {pred['ipc_code']} - {pred['percentage']}\n"
                explanation += f"   {pred['description']}\n"
            
            explanation += "\n" + "-" * 60 + "\n"
            explanation += "Top 10 Most Important Features:\n"
            explanation += "-" * 60 + "\n"
            for feat in details['important_features']:
                explanation += f"{feat['rank']:2d}. {feat['feature_name']:30s} "
                explanation += f"{feat['importance_percentage']}\n"
        
        return explanation
    
    def confidence_above_threshold(self):
        """Check if last prediction confidence is above threshold"""
        if self.last_prediction_details is None:
            return False
        
        confidence = self.last_prediction_details.get('confidence', 0)
        return confidence >= CONFIDENCE_THRESHOLD
    
    def get_model_info(self):
        """Get information about the trained model"""
        info = "MODEL INFORMATION\n"
        info += "=" * 60 + "\n\n"
        
        info += f"Number of classes (IPC sections): {len(self.label_decoder)}\n"
        info += f"Number of features: {len(self.feature_names)}\n"
        info += f"Number of trees: {self.model.n_estimators}\n"
        info += f"Max depth: {self.model.max_depth}\n\n"
        
        info += "IPC Sections:\n"
        info += "-" * 60 + "\n"
        for ipc_code, description in self.ipc_sections.items():
            info += f"{ipc_code}: {description}\n"
        
        return info


def predict_from_file(image_path):
    """Utility function to predict from file"""
    predictor = IPCPredictor()
    result = predictor.predict_single(image_path)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    
    print("\n" + predictor.get_prediction_explanation())
    
    return result


def main():
    """Demo prediction system"""
    try:
        predictor = IPCPredictor()
        
        print(predictor.get_model_info())
        
        print("\nNote: To make predictions, provide an image file or use predict_from_file()")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train the model first by running: python scripts/training.py")


if __name__ == "__main__":
    main()
