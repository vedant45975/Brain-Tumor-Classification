"""Detailed test showing metrics for each prediction"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image

TEST_IMAGES = [
    (r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\glioma\Te-glTr_0000.jpg", "glioma"),
    (r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\meningioma\Te-meTr_0000.jpg", "meningioma"),
    (r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\notumor\Te-noTr_0000.jpg", "notumor"),
    (r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\pituitary\Te-piTr_0000.jpg", "pituitary"),
]

from app.main import predict_with_fallback

print("\n" + "="*90)
print("DETAILED PREDICTION ANALYSIS")
print("="*90)

for img_path, expected in TEST_IMAGES:
    img = Image.open(img_path)
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_batch = np.expand_dims(img_array, axis=0)
    predicted, confidence = predict_with_fallback(img_batch)
    
    # Get metrics
    flat = img_array.flatten()
    mean_intensity = np.mean(flat)
    entropy = -np.sum(np.histogram(flat, bins=10)[0] / len(flat) * np.log2(np.maximum(np.histogram(flat, bins=10)[0] / len(flat), 1e-10)))
    diffs = np.abs(np.diff(flat))
    edge_strength = np.sum(diffs > 0.05) / len(diffs)
    
    status = "✅" if predicted == expected else "❌"
    print(f"\n{status} {expected.upper()}: Predicted {predicted.upper()} ({confidence}%)")
    print(f"   Mean: {mean_intensity:.4f} | Entropy: {entropy:.4f} | EdgeStr: {edge_strength:.4f}")

print("\n" + "="*90 + "\n")
