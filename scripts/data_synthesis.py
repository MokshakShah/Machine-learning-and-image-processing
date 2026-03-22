"""
Synthetic FIR Document Image Generator
Generates realistic noisy FIR document images for training the IPC classifier
"""

import numpy as np
import cv2
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
from config import (
    SYNTHETIC_DATA_DIR, IPC_SECTIONS, SYNTHETIC_IMAGE_SIZE,
    NOISE_LEVELS, BLUR_KERNELS, ROTATION_ANGLES, PERSPECTIVE_DISTORTION_RANGE,
    SYNTHETIC_SAMPLES_PER_IPC
)


class SyntheticFIRGenerator:
    """Generate synthetic FIR document images with IPC sections"""
    
    def __init__(self):
        self.image_size = SYNTHETIC_IMAGE_SIZE
        self.ipc_sections = IPC_SECTIONS
        self.samples_per_ipc = SYNTHETIC_SAMPLES_PER_IPC
        
    def generate_fir_background(self):
        """Create realistic FIR document background with header and text"""
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to load a default font, fallback to default
        try:
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            header_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw FIR header
        draw.text((20, 20), "FIRST INFORMATION REPORT (F.I.R.)", font=header_font, fill='black')
        draw.line([(20, 50), (self.image_size[0]-20, 50)], fill='black', width=2)
        
        # Draw FIR details
        y_pos = 70
        details = [
            "Station: City Police Station",
            "Date: DD/MM/YYYY",
            "Time: HH:MM",
            "Complaint Type: Crime Report",
        ]
        
        for detail in details:
            draw.text((30, y_pos), detail, font=body_font, fill='black')
            y_pos += 30
        
        # Add some random case details
        y_pos += 20
        draw.text((30, y_pos), "Case Details:", font=header_font, fill='black')
        y_pos += 30
        
        case_texts = [
            "Incident Description: [INCIDENT DETAILS]",
            "Location: [ADDRESS]",
            "Complainant: [NAME]",
            "Accused: [NAME]",
        ]
        
        for text in case_texts:
            draw.text((40, y_pos), text, font=body_font, fill='black')
            y_pos += 25
        
        return np.array(img)
    
    def add_noise(self, img, noise_level):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_level * 255, img.shape)
        noisy_img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        return noisy_img
    
    def add_blur(self, img, kernel_size):
        """Add Gaussian blur to simulate document scan quality"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def add_rotation(self, img, angle):
        """Rotate image to simulate document scanning angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
        return rotated
    
    def add_perspective_distortion(self, img):
        """Add perspective distortion to simulate scanning from angle"""
        h, w = img.shape[:2]
        
        # Random distortion
        dist = int(self.image_size[0] * PERSPECTIVE_DISTORTION_RANGE)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.randint(-dist, dist), random.randint(-dist, dist)],
            [w + random.randint(-dist, dist), random.randint(-dist, dist)],
            [random.randint(-dist, dist), h + random.randint(-dist, dist)],
            [w + random.randint(-dist, dist), h + random.randint(-dist, dist)]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        distorted = cv2.warpPerspective(img, M, (w, h), borderValue=255)
        return distorted
    
    def embed_ipc_section(self, img, ipc_code):
        """Embed IPC section code in the document"""
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Place IPC code at bottom with some randomization
        y_offset = self.image_size[0] - 80 + random.randint(-10, 10)
        x_offset = random.randint(30, 100)
        
        text = f"IPC Section: {ipc_code}"
        draw.text((x_offset, y_offset), text, font=font, fill='black')
        
        return np.array(img_pil)
    
    def add_scan_artifacts(self, img):
        """Add realistic scan artifacts like shadows, dust, etc."""
        # Add slight shadow effect
        h, w = img.shape[:2]
        shadow = np.ones_like(img, dtype=float) * 255
        
        for i in range(h):
            intensity = 255 - (i / h) * 30  # Gradient shadow
            shadow[i] *= intensity / 255
        
        img = np.clip(img.astype(float) * (shadow / 255), 0, 255).astype(np.uint8)
        
        # Add random dust spots
        for _ in range(random.randint(5, 15)):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(2, 6)
            cv2.circle(img, (x, y), size, (200, 200, 200), -1)
        
        return img
    
    def generate_single_sample(self, ipc_code):
        """Generate a single synthetic FIR image with augmentation"""
        # Start with clean FIR background
        img = self.generate_fir_background()
        
        # Embed IPC section
        img = self.embed_ipc_section(img, ipc_code)
        
        # Add augmentations
        noise_level = random.choice(NOISE_LEVELS)
        img = self.add_noise(img, noise_level)
        
        # Blur
        blur_kernel = random.choice(BLUR_KERNELS)
        img = self.add_blur(img, blur_kernel)
        
        # Rotation (small angles to keep text somewhat readable)
        angle = random.choice(ROTATION_ANGLES)
        img = self.add_rotation(img, angle)
        
        # Perspective distortion
        if random.random() > 0.5:
            img = self.add_perspective_distortion(img)
        
        # Scan artifacts
        img = self.add_scan_artifacts(img)
        
        # Convert to grayscale (typical document)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return img
    
    def generate_dataset(self):
        """Generate complete synthetic dataset for all IPC sections"""
        print(f"Generating synthetic FIR dataset...")
        print(f"Total samples: {len(self.ipc_sections) * self.samples_per_ipc}")
        
        dataset_info = []
        sample_count = 0
        
        for ipc_code, ipc_desc in self.ipc_sections.items():
            print(f"\nGenerating samples for {ipc_code}...")
            
            for i in range(self.samples_per_ipc):
                # Generate image
                img = self.generate_single_sample(ipc_code)
                
                # Save image
                filename = f"{ipc_code}_{i:03d}.png"
                filepath = SYNTHETIC_DATA_DIR / filename
                cv2.imwrite(str(filepath), img)
                
                dataset_info.append({
                    'filename': filename,
                    'ipc_code': ipc_code,
                    'ipc_description': ipc_desc,
                    'filepath': str(filepath)
                })
                
                sample_count += 1
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{self.samples_per_ipc} samples")
        
        print(f"\nDataset generation complete!")
        print(f"Total samples generated: {sample_count}")
        print(f"Saved to: {SYNTHETIC_DATA_DIR}")
        
        return dataset_info


def main():
    """Generate synthetic dataset"""
    generator = SyntheticFIRGenerator()
    dataset_info = generator.generate_dataset()
    
    # Save dataset info
    import json
    info_path = SYNTHETIC_DATA_DIR / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset info saved to: {info_path}")


if __name__ == "__main__":
    main()
