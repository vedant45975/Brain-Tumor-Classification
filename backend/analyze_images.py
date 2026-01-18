"""
Debug script to analyze image characteristics
"""
import cv2
import numpy as np
from pathlib import Path

TEST_DIR = r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test"

def analyze_image(image_path):
    """Analyze image and print characteristics"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return
    
    img = cv2.resize(img, (160, 160))
    img = img / 255.0
    
    img = img[:, :, ::-1]  # Convert BGR to RGB
    
    flat = img.flatten()
    
    mean_intensity = np.mean(flat)
    std_intensity = np.std(flat)
    median_intensity = np.median(flat)
    min_intensity = np.min(flat)
    max_intensity = np.max(flat)
    contrast = max_intensity - min_intensity
    
    diffs = np.abs(np.diff(flat))
    edge_strength = np.sum(diffs > 0.05) / len(diffs)
    edge_mean = np.mean(diffs)
    
    hist, _ = np.histogram(flat, bins=10)
    hist_norm = hist / np.sum(hist)
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + 1e-10))
    
    h, w = img.shape[:2]
    center = img[h//4:3*h//4, w//4:3*w//4]
    corners = np.concatenate([
        img[:h//4, :w//4].flatten(),
        img[:h//4, 3*w//4:].flatten(),
        img[3*h//4:, :w//4].flatten(),
        img[3*h//4:, 3*w//4:].flatten()
    ])
    
    center_mean = np.mean(center)
    corners_mean = np.mean(corners) if len(corners) > 0 else 0
    center_concentration = center_mean - corners_mean
    
    print(f"\nImage: {Path(image_path).name}")
    print(f"{'─'*60}")
    print(f"Mean Intensity:      {mean_intensity:.4f}")
    print(f"Std Intensity:       {std_intensity:.4f}")
    print(f"Median Intensity:    {median_intensity:.4f}")
    print(f"Contrast (max-min):  {contrast:.4f}")
    print(f"Edge Strength:       {edge_strength:.4f}")
    print(f"Edge Mean:           {edge_mean:.4f}")
    print(f"Entropy:             {entropy:.4f}")
    print(f"Center Concentration: {center_concentration:.4f}")

print("\n" + "="*60)
print("IMAGE ANALYSIS")
print("="*60)

# Analyze one image from each class
analyze_image(r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\glioma\Te-gl_0010.jpg")
analyze_image(r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\meningioma\Te-me_0011.jpg")
analyze_image(r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-no_0010.jpg")
analyze_image(r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\pituitary\Te-pi_0015.jpg")

print("\n" + "="*60)
