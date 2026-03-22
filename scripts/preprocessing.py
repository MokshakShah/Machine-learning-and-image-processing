"""
Image Processing and Computer Vision (IPCV) Pipeline
6-stage preprocessing for document image enhancement and normalization
"""

import cv2
import numpy as np
from pathlib import Path
from config import (
    CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH,
    SYNTHETIC_IMAGE_SIZE
)


class IPCVPreprocessor:
    """
    6-stage IPCV pipeline for forensic document processing:
    1. Noise Reduction
    2. Contrast Enhancement
    3. Perspective Correction
    4. Binarization
    5. Morphological Operations
    6. Edge Detection & Analysis
    """
    
    def __init__(self, target_size=SYNTHETIC_IMAGE_SIZE):
        self.target_size = target_size
        self.processing_steps = []

    def _to_uint8_gray(self, img):
        """Normalize input image to single-channel uint8."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _entropy(self, img):
        """Compute Shannon entropy of grayscale intensities."""
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        prob = hist / (hist.sum() + 1e-8)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob + 1e-12)))

    def analyze_image_quality(self, img):
        """Return interpretable quality diagnostics for an input image."""
        gray = self._to_uint8_gray(img)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_std = float(np.std(gray))
        mean_brightness = float(np.mean(gray))
        entropy = self._entropy(gray)

        sharpness_score = min(max(lap_var / 150.0, 0.0), 1.0)
        contrast_score = min(max(contrast_std / 64.0, 0.0), 1.0)
        brightness_score = max(0.0, 1.0 - (abs(mean_brightness - 127.0) / 127.0))
        entropy_score = min(max(entropy / 8.0, 0.0), 1.0)

        quality_score = (
            0.35 * sharpness_score
            + 0.30 * contrast_score
            + 0.20 * entropy_score
            + 0.15 * brightness_score
        )

        return {
            "laplacian_variance": lap_var,
            "contrast_std": contrast_std,
            "mean_brightness": mean_brightness,
            "entropy": entropy,
            "sharpness_score": float(sharpness_score),
            "contrast_score": float(contrast_score),
            "brightness_score": float(brightness_score),
            "entropy_score": float(entropy_score),
            "quality_score": float(quality_score),
        }

    def enhance_readability(self, img):
        """Create a readability-optimized variant for difficult document images."""
        gray = self._to_uint8_gray(img)

        # Upscale lightly to preserve thin strokes in low-resolution captures.
        h, w = gray.shape[:2]
        if min(h, w) < 800:
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

        return self._to_uint8_gray(sharpened)

    def suppress_stains(self, img):
        """Reduce paper stains and background texture using background normalization."""
        gray = self._to_uint8_gray(img)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        normalized = cv2.divide(gray, background, scale=255)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)
        return self._to_uint8_gray(normalized)

    def deblur_text(self, img):
        """Strengthen text strokes for mildly blurred captures."""
        gray = self._to_uint8_gray(img)
        denoised = cv2.fastNlMeansDenoising(gray, h=6, templateWindowSize=7, searchWindowSize=21)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        sharpened = cv2.addWeighted(denoised, 1.55, gaussian, -0.55, 0)
        return self._to_uint8_gray(sharpened)

    def generate_enhancement_variants(self, img):
        """Generate multiple enhanced views to improve downstream model robustness."""
        gray = self._to_uint8_gray(img)
        readability = self.enhance_readability(gray)
        stain_reduced = self.suppress_stains(readability)
        deblurred = self.deblur_text(stain_reduced)
        deskewed = self.stage3_perspective_correction(deblurred.copy())

        return {
            "original": gray,
            "readability": readability,
            "stain_reduced": stain_reduced,
            "deblurred": deblurred,
            "deskewed": deskewed,
        }

    def get_best_refined_image(self, img):
        """Return the best refined variant based on quality score."""
        variants = self.generate_enhancement_variants(img)
        best_name = "original"
        best_img = variants[best_name]
        best_quality = self.analyze_image_quality(best_img)

        for name, variant in variants.items():
            quality = self.analyze_image_quality(variant)
            if quality.get("quality_score", 0.0) > best_quality.get("quality_score", 0.0):
                best_name = name
                best_img = variant
                best_quality = quality

        return best_name, best_img, best_quality, variants
    
    def stage1_noise_reduction(self, img):
        """
        Stage 1: Advanced noise reduction
        Uses Non-Local Means Denoising for document quality preservation
        """
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Apply Non-Local Means Denoising
        denoised = cv2.fastNlMeansDenoising(
            img,
            h=10,  # Filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        self.processing_steps.append(("Noise Reduction", denoised.copy()))
        return denoised
    
    def stage2_contrast_enhancement(self, img):
        """
        Stage 2: Adaptive Histogram Equalization
        Improves contrast locally without over-enhancement
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        self.processing_steps.append(("Contrast Enhancement", enhanced.copy()))
        return enhanced
    
    def stage3_perspective_correction(self, img):
        """
        Stage 3: Perspective correction via edge-based document detection
        Detects document boundaries and corrects skew
        """
        # Find document edges using Canny
        edges = cv2.Canny(img, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find largest contour (likely the document)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(largest_contour)
            
            # Only apply perspective if rotation is significant
            angle = rect[2]
            if abs(angle) > 2:
                # Correct rotation
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                corrected = cv2.warpAffine(img, M, (w, h), borderValue=255)
                img = corrected
        
        self.processing_steps.append(("Perspective Correction", img.copy()))
        return img
    
    def stage4_binarization(self, img):
        """
        Stage 4: Adaptive Thresholding for document binarization
        Handles variable lighting conditions
        """
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        self.processing_steps.append(("Binarization", binary.copy()))
        return binary
    
    def stage5_morphological_operations(self, img):
        """
        Stage 5: Morphological operations to clean up binary image
        Remove noise and connect broken characters
        """
        # Create morphological kernels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Close operation (fill small holes)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Open operation (remove small noise)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Dilate to connect broken characters
        dilated = cv2.dilate(opened, kernel_small, iterations=1)
        
        self.processing_steps.append(("Morphological Ops", dilated.copy()))
        return dilated
    
    def stage6_edge_detection(self, img):
        """
        Stage 6: Multi-scale edge detection for feature extraction
        Detects document structure and text regions
        """
        # Canny edge detection
        edges = cv2.Canny(img, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        
        # Dilate edges slightly to strengthen them
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        self.processing_steps.append(("Edge Detection", edges.copy()))
        return edges
    
    def process(self, img_path_or_array, return_intermediate=False):
        """
        Execute complete 6-stage IPCV pipeline
        
        Args:
            img_path_or_array: Path to image or numpy array
            return_intermediate: If True, return all intermediate stages
        
        Returns:
            Processed image (or dict of all stages if return_intermediate=True)
        """
        # Load image if path provided
        if isinstance(img_path_or_array, (str, Path)):
            img = cv2.imread(str(img_path_or_array), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {img_path_or_array}")
        else:
            img = img_path_or_array.copy()
        
        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        img = cv2.resize(img, self.target_size)
        
        # Clear previous steps
        self.processing_steps = []
        self.processing_steps.append(("Original", img.copy()))
        
        # Execute pipeline stages
        img = self.stage1_noise_reduction(img)
        img = self.stage2_contrast_enhancement(img)
        img = self.stage3_perspective_correction(img)
        img = self.stage4_binarization(img)
        img = self.stage5_morphological_operations(img)
        img = self.stage6_edge_detection(img)
        
        if return_intermediate:
            return {stage_name: stage_img for stage_name, stage_img in self.processing_steps}
        
        return img
    
    def get_processing_report(self):
        """Get detailed report of all processing stages"""
        report = "IPCV Processing Report\n"
        report += "=" * 50 + "\n"
        
        for i, (stage_name, stage_img) in enumerate(self.processing_steps, 1):
            report += f"\nStage {i}: {stage_name}\n"
            report += f"  Shape: {stage_img.shape}\n"
            report += f"  Data type: {stage_img.dtype}\n"
            report += f"  Min value: {stage_img.min()}\n"
            report += f"  Max value: {stage_img.max()}\n"
            report += f"  Mean value: {stage_img.mean():.2f}\n"
        
        return report


def preprocess_directory(input_dir, output_dir):
    """Preprocess all images in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocessor = IPCVPreprocessor()
    
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    
    for img_file in image_files:
        try:
            processed = preprocessor.process(img_file)
            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), processed)
            print(f"Processed: {img_file.name}")
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    print(f"All images processed. Output saved to: {output_path}")


def main():
    """Test preprocessing pipeline"""
    preprocessor = IPCVPreprocessor()
    
    # Create a test image
    test_img = np.random.randint(0, 255, (400, 600), dtype=np.uint8)
    
    # Process it
    result = preprocessor.process(test_img, return_intermediate=True)
    
    # Print report
    print(preprocessor.get_processing_report())


if __name__ == "__main__":
    main()
