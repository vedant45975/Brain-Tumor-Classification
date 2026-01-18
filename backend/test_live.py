"""Test using the LIVE algorithm from main.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image

# Import the actual prediction function from main
from app.main import predict_with_fallback

TEST_IMAGES = {
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

print("\n================================================================================")
print("LIVE ALGORITHM TEST (imports from app.main.predict_with_fallback)")
print("================================================================================\n")

results = {"glioma": 0, "meningioma": 0, "notumor": 0, "pituitary": 0}
total = 0

for expected_type, image_paths in TEST_IMAGES.items():
    print(f"Testing {expected_type.upper()}:")
    correct = 0
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img = img.resize((160, 160))
            img_array = np.array(img) / 255.0
            
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            img_batch = np.expand_dims(img_array, axis=0)
            predicted_type, confidence = predict_with_fallback(img_batch)
            
            is_correct = predicted_type == expected_type
            if is_correct:
                correct += 1
                results[expected_type] += 1
            total += 1
            
            status = "OK" if is_correct else "X"
            print(f"  [{status}] Expected: {expected_type:12} Got: {predicted_type:12} ({confidence}%)")
            
        except Exception as e:
            print(f"  [X] Error: {str(e)[:60]}")
            total += 1
    
    print(f"  Result: {correct}/3 correct\n")

print("================================================================================")
print("SUMMARY")
print("================================================================================")
for tumor_type in ["glioma", "meningioma", "notumor", "pituitary"]:
    pct = (results[tumor_type] / 3) * 100
    print(f"{tumor_type:12}: {results[tumor_type]}/3 ({pct:5.1f}%)")

overall_pct = (sum(results.values()) / total) * 100 if total > 0 else 0
print(f"{'-'*40}")
print(f"{'OVERALL':12}: {sum(results.values())}/{total} ({overall_pct:5.1f}%)")
print("="*80 + "\n")
