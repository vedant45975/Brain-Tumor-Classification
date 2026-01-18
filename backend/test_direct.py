"""Direct test of fallback prediction algorithm - no server needed"""
import sys
import os
import io
import numpy as np
from PIL import Image

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the prediction function directly
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

print("\n" + "="*80)
print("DIRECT PREDICTION TEST (No Server Required)")
print("="*80)

results = {"glioma": 0, "meningioma": 0, "notumor": 0, "pituitary": 0}
total = 0

for expected_type, image_paths in TEST_IMAGES.items():
    print(f"\n{'-'*80}")
    print(f"Testing {expected_type.upper()} images ({len(image_paths)} samples):")
    print(f"{'-'*80}")
    
    correct = 0
    for img_path in image_paths:
        try:
            # Load and preprocess image
            img = Image.open(img_path)
            img = img.resize((160, 160))
            img_array = np.array(img) / 255.0
            
            # Ensure 3D array (add batch dimension)
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Get prediction
            predicted_type, confidence = predict_with_fallback(img_batch)
            
            is_correct = predicted_type == expected_type
            status = "✅" if is_correct else "❌"
            
            if is_correct:
                correct += 1
                results[expected_type] += 1
            
            total += 1
            
            print(f"{status} Expected: {expected_type:12} | Predicted: {predicted_type:12} | Confidence: {confidence}%")
            
        except Exception as e:
            print(f"❌ Error testing {img_path}: {e}")
            total += 1
    
    print(f"\n{expected_type.upper()}: {correct}/{len(image_paths)} correct")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
for tumor_type, count in results.items():
    pct = (count / 3) * 100 if total > 0 else 0
    print(f"{tumor_type:12} | {count}/3 correct ({pct:5.1f}%)")

overall_pct = (sum(results.values()) / total) * 100 if total > 0 else 0
print(f"{'-'*30}")
print(f"{'OVERALL':12} | {sum(results.values())}/{total} correct ({overall_pct:5.1f}%)")
print("="*80 + "\n")
