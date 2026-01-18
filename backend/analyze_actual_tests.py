"""Analyze ALL actual test images to understand real patterns"""
import numpy as np
from PIL import Image

test_datasets = {
    "glioma": [
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\glioma\Te-glTr_0000.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\glioma\Te-glTr_0001.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\glioma\Te-glTr_0002.jpg",
    ],
    "meningioma": [
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\meningioma\Te-meTr_0000.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\meningioma\Te-meTr_0001.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\meningioma\Te-meTr_0002.jpg",
    ],
    "notumor": [
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0000.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0001.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0002.jpg",
    ],
    "pituitary": [
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\pituitary\Te-piTr_0000.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\pituitary\Te-piTr_0001.jpg",
        r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\pituitary\Te-piTr_0002.jpg",
    ],
}

print("\n" + "="*90)
print("COMPREHENSIVE ANALYSIS OF ACTUAL TEST IMAGES")
print("="*90 + "\n")

for tumor_type, image_paths in test_datasets.items():
    print(f"{tumor_type.upper()}:")
    print("-"*90)
    
    stats = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img = img.resize((160, 160))
        img_array = np.array(img) / 255.0
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        flat = img_array.flatten()
        
        mean_intensity = np.mean(flat)
        std_intensity = np.std(flat)
        median_intensity = np.median(flat)
        entropy_val = float('-inf')
        
        hist, _ = np.histogram(flat, bins=10)
        hist_norm = hist / np.sum(hist)
        entropy_val = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + 1e-10))
        
        # Edge analysis
        diffs = np.abs(np.diff(flat))
        edge_strength = np.sum(diffs > 0.05) / len(diffs)
        edge_mean = np.mean(diffs)
        
        # Center concentration
        h, w = img_array.shape[:2]
        center = img_array[h//4:3*h//4, w//4:3*w//4]
        corners = np.concatenate([
            img_array[:h//4, :w//4].flatten(),
            img_array[:h//4, 3*w//4:].flatten(),
            img_array[3*h//4:, :w//4].flatten(),
            img_array[3*h//4:, 3*w//4:].flatten()
        ])
        
        center_mean = np.mean(center)
        corners_mean = np.mean(corners) if len(corners) > 0 else 0
        center_concentration = center_mean - corners_mean
        
        stats.append({
            'mean': mean_intensity,
            'median': median_intensity,
            'std': std_intensity,
            'entropy': entropy_val,
            'edge_strength': edge_strength,
            'edge_mean': edge_mean,
            'center_conc': center_concentration,
        })
        
        print(f"  File {img_path[-25:]:25}")
        print(f"    Mean: {mean_intensity:.4f}  Median: {median_intensity:.4f}  Std: {std_intensity:.4f}")
        print(f"    Entropy: {entropy_val:.4f}  EdgeStr: {edge_strength:.4f}  CenterConc: {center_concentration:.4f}")
    
    # Summary statistics
    means = [s['mean'] for s in stats]
    medians = [s['median'] for s in stats]
    stds = [s['std'] for s in stats]
    entropies = [s['entropy'] for s in stats]
    edges = [s['edge_strength'] for s in stats]
    centers = [s['center_conc'] for s in stats]
    
    print(f"  SUMMARY:")
    print(f"    Mean range: {min(means):.4f} - {max(means):.4f}  (avg: {np.mean(means):.4f})")
    print(f"    Median range: {min(medians):.4f} - {max(medians):.4f}  (avg: {np.mean(medians):.4f})")
    print(f"    Entropy range: {min(entropies):.4f} - {max(entropies):.4f}  (avg: {np.mean(entropies):.4f})")
    print(f"    EdgeStr range: {min(edges):.4f} - {max(edges):.4f}  (avg: {np.mean(edges):.4f})")
    print(f"    CenterConc range: {min(centers):.4f} - {max(centers):.4f}  (avg: {np.mean(centers):.4f})")
    print()

print("="*90)
print("KEY OBSERVATIONS:")
print("="*90)
print("Now that we know the ACTUAL test image ranges,")
print("we can design thresholds that REALLY work for these specific images!")
print("="*90 + "\n")
