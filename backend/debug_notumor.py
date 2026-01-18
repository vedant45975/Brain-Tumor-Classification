"""Debug why notumor images are failing"""
import numpy as np
from PIL import Image

notumor_images = [
    r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0000.jpg",
    r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0001.jpg",
    r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0002.jpg",
]

print("\nNOTUMOR IMAGES ANALYSIS:")
print("="*80)

for img_path in notumor_images:
    img = Image.open(img_path)
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    flat = img_array.flatten()
    
    mean_intensity = np.mean(flat)
    median_intensity = np.median(flat)
    center_concentration = 0.1417  # example
    entropy = np.sum(-hist_norm * np.log2(hist_norm + 1e-10)) if 'hist_norm' in locals() else 2.47
    
    print(f"\nFile: {img_path.split('/')[-1]}")
    print(f"  Mean: {mean_intensity:.4f} (target > 0.30)")
    print(f"  Median: {median_intensity:.4f} (target > 0.25)")
    print(f"  Entropy:", end=" ")
    
    hist, _ = np.histogram(flat, bins=10)
    hist_norm = hist / np.sum(hist)
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + 1e-10))
    print(f"{entropy:.4f} (target > 2.35)")
    
    # Check what scores this would get
    notumor_score = 0
    if mean_intensity > 0.30:
        notumor_score += 50
        print(f"  ✓ Mean score: +50 = {notumor_score}")
    else:
        print(f"  ✗ Mean {mean_intensity:.4f} NOT > 0.30")
    
    if median_intensity > 0.25:
        notumor_score += 45
        print(f"  ✓ Median score: +45 = {notumor_score}")
    else:
        print(f"  ✗ Median {median_intensity:.4f} NOT > 0.25")
    
    if entropy > 2.35:
        notumor_score += 35
        print(f"  ✓ Entropy score: +35 = {notumor_score}")
    else:
        print(f"  ✗ Entropy {entropy:.4f} NOT > 2.35")
    
    print(f"  NOTUMOR TOTAL SCORE: {notumor_score}")
