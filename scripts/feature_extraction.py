"""
Feature Extraction Module
Extracts 65+ texture and edge-based features from preprocessed document images
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from scipy import ndimage
from pathlib import Path
from preprocessing import IPCVPreprocessor
from config import (
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    LBP_RADIUS, LBP_POINTS
)


class FeatureExtractor:
    """
    Extract comprehensive feature set from document images:
    - Texture features (LBP, HOG)
    - Edge-based features (Canny, Sobel)
    - Contour features
    - Statistical features
    - Text region features (via morphology)
    """
    
    def __init__(self):
        self.preprocessor = IPCVPreprocessor()
        self.feature_names = []
        self.feature_vector = None
    
    def extract_hog_features(self, img):
        """
        Extract Histogram of Oriented Gradients (HOG) features
        Captures edge orientation and distribution
        """
        hog_features = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=(HOG_PIXELS_PER_CELL, HOG_PIXELS_PER_CELL),
            cells_per_block=(HOG_CELLS_PER_BLOCK, HOG_CELLS_PER_BLOCK),
            feature_vector=True
        )
        
        # Normalize
        hog_features = (hog_features - hog_features.mean()) / (hog_features.std() + 1e-6)
        
        return hog_features
    
    def extract_lbp_features(self, img):
        """
        Extract Local Binary Pattern (LBP) features
        Captures local texture patterns
        """
        lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method='uniform')
        
        # Create histogram of LBP values
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, LBP_POINTS + 3),
            range=(0, LBP_POINTS + 2)
        )
        
        # Normalize histogram
        hist = hist.astype(float) / hist.sum()
        
        return hist
    
    def extract_edge_features(self, img):
        """
        Extract edge-based features using Sobel operators
        Captures gradient information
        """
        # Sobel derivatives
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Statistical features from magnitude
        features = np.array([
            np.mean(magnitude),
            np.std(magnitude),
            np.min(magnitude),
            np.max(magnitude),
            np.percentile(magnitude, 25),
            np.percentile(magnitude, 50),
            np.percentile(magnitude, 75),
        ])
        
        return features
    
    def extract_contour_features(self, img):
        """
        Extract features based on contours (document structure)
        Captures text regions and document layout
        """
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        # Number of contours
        features.append(len(contours))
        
        if len(contours) > 0:
            # Contour areas
            areas = [cv2.contourArea(c) for c in contours]
            features.extend([
                np.mean(areas),
                np.std(areas),
                np.min(areas),
                np.max(areas),
                np.sum(areas)
            ])
            
            # Contour perimeters
            perimeters = [cv2.arcLength(c, True) for c in contours]
            features.extend([
                np.mean(perimeters),
                np.std(perimeters),
                np.max(perimeters)
            ])
            
            # Circularity (how round the contours are)
            circularities = []
            for area, perimeter in zip(areas, perimeters):
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    circularities.append(circularity)
            
            if circularities:
                features.extend([
                    np.mean(circularities),
                    np.std(circularities),
                ])
        else:
            # Default features if no contours found
            features.extend([0] * 10)
        
        return np.array(features)
    
    def extract_statistical_features(self, img):
        """Extract basic statistical features from image"""
        features = np.array([
            np.mean(img),
            np.std(img),
            np.min(img),
            np.max(img),
            np.percentile(img, 25),
            np.percentile(img, 50),
            np.percentile(img, 75),
            np.percentile(img, 90),
            np.percentile(img, 95),
            skewness_measure(img),
            kurtosis_measure(img),
        ])
        
        return features
    
    def extract_morphological_features(self, img):
        """Extract features from morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Erosion
        eroded = cv2.erode(img, kernel, iterations=1)
        
        # Dilation
        dilated = cv2.dilate(img, kernel, iterations=1)
        
        # Opening
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        # Closing
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        features = np.array([
            np.mean(eroded),
            np.mean(dilated),
            np.mean(opened),
            np.mean(closed),
            np.std(eroded),
            np.std(dilated),
            np.std(opened),
            np.std(closed),
        ])
        
        return features
    
    def extract_text_region_features(self, img):
        """
        Extract features related to text regions
        Assumes binarized input
        """
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=8
        )
        
        features = []
        
        # Number of connected components (potential text regions)
        features.append(num_labels)
        
        if num_labels > 1:  # Exclude background
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            features.extend([
                np.mean(component_areas),
                np.std(component_areas),
                np.min(component_areas),
                np.max(component_areas),
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Text density (ratio of text pixels to total)
        text_density = np.sum(img > 128) / img.size
        features.append(text_density)
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, img):
        """Extract features from frequency domain (FFT)"""
        # Compute FFT
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        features = np.array([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.max(magnitude_spectrum),
            np.percentile(magnitude_spectrum, 90),
        ])
        
        return features
    
    def extract_all_features(self, img_path_or_array):
        """
        Extract complete feature vector from image (65+ features)
        
        Args:
            img_path_or_array: Path to image or numpy array
        
        Returns:
            Feature vector (numpy array) and feature names
        """
        # Preprocess image
        processed_stages = self.preprocessor.process(
            img_path_or_array,
            return_intermediate=True
        )
        
        # Use the final processed image (edge detection)
        processed_img = list(processed_stages.values())[-1]
        
        # Also get the binarized version for text features
        binary_img = processed_stages.get("Binarization", processed_img)
        
        features = []
        feature_names = []
        
        # 1. HOG Features (~180 features depending on image size)
        hog_feat = self.extract_hog_features(processed_img)
        features.append(hog_feat)
        feature_names.extend([f"HOG_{i}" for i in range(len(hog_feat))])
        
        # 2. LBP Features (typically ~59 features)
        lbp_feat = self.extract_lbp_features(processed_img)
        features.append(lbp_feat)
        feature_names.extend([f"LBP_{i}" for i in range(len(lbp_feat))])
        
        # 3. Edge Features (7 features)
        edge_feat = self.extract_edge_features(processed_img)
        features.append(edge_feat)
        feature_names.extend([
            "Edge_Mean", "Edge_Std", "Edge_Min", "Edge_Max",
            "Edge_Q25", "Edge_Q50", "Edge_Q75"
        ])
        
        # 4. Contour Features (~13 features)
        contour_feat = self.extract_contour_features(processed_img)
        features.append(contour_feat)
        feature_names.extend([
            "Contour_Count", "Contour_Area_Mean", "Contour_Area_Std",
            "Contour_Area_Min", "Contour_Area_Max", "Contour_Area_Sum",
            "Contour_Perim_Mean", "Contour_Perim_Std", "Contour_Perim_Max",
            "Circularity_Mean", "Circularity_Std"
        ])
        
        # 5. Statistical Features (11 features)
        stat_feat = self.extract_statistical_features(processed_img)
        features.append(stat_feat)
        feature_names.extend([
            "Stat_Mean", "Stat_Std", "Stat_Min", "Stat_Max",
            "Stat_Q25", "Stat_Q50", "Stat_Q75", "Stat_Q90", "Stat_Q95",
            "Skewness", "Kurtosis"
        ])
        
        # 6. Morphological Features (8 features)
        morph_feat = self.extract_morphological_features(processed_img)
        features.append(morph_feat)
        feature_names.extend([
            "Morph_Erode_Mean", "Morph_Dilate_Mean", "Morph_Open_Mean", "Morph_Close_Mean",
            "Morph_Erode_Std", "Morph_Dilate_Std", "Morph_Open_Std", "Morph_Close_Std"
        ])
        
        # 7. Text Region Features (6 features)
        text_feat = self.extract_text_region_features(binary_img)
        features.append(text_feat)
        feature_names.extend([
            "Text_Components", "Text_Area_Mean", "Text_Area_Std",
            "Text_Area_Min", "Text_Area_Max", "Text_Density"
        ])
        
        # 8. Frequency Domain Features (4 features)
        freq_feat = self.extract_frequency_domain_features(processed_img)
        features.append(freq_feat)
        feature_names.extend([
            "Freq_Mean", "Freq_Std", "Freq_Max", "Freq_Q90"
        ])
        
        # Concatenate all features
        feature_vector = np.concatenate(features)
        
        self.feature_names = feature_names
        self.feature_vector = feature_vector
        
        return feature_vector, feature_names
    
    def get_feature_summary(self):
        """Get summary statistics of extracted features"""
        if self.feature_vector is None:
            return "No features extracted yet"
        
        summary = "Feature Extraction Summary\n"
        summary += "=" * 50 + "\n"
        summary += f"Total features: {len(self.feature_vector)}\n"
        summary += f"Feature mean: {np.mean(self.feature_vector):.4f}\n"
        summary += f"Feature std: {np.std(self.feature_vector):.4f}\n"
        summary += f"Feature min: {np.min(self.feature_vector):.4f}\n"
        summary += f"Feature max: {np.max(self.feature_vector):.4f}\n"
        
        return summary


def skewness_measure(img):
    """Calculate skewness of image intensity"""
    flat = img.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    if std == 0:
        return 0
    return np.mean(((flat - mean) / std) ** 3)


def kurtosis_measure(img):
    """Calculate kurtosis of image intensity"""
    flat = img.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    if std == 0:
        return 0
    return np.mean(((flat - mean) / std) ** 4) - 3


def main():
    """Test feature extraction"""
    extractor = FeatureExtractor()
    
    # Create test image
    test_img = np.random.randint(0, 255, (400, 600), dtype=np.uint8)
    
    # Extract features
    features, feature_names = extractor.extract_all_features(test_img)
    
    print(f"Extracted {len(features)} features")
    print(f"Feature names: {len(feature_names)}")
    print(extractor.get_feature_summary())


if __name__ == "__main__":
    main()
