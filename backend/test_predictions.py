"""
Test script to validate predictions on test images
"""
import requests
import os
from pathlib import Path

# Backend API endpoint
API_URL = "http://localhost:8000"

# Test images directory
TEST_DIR = r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test"

def test_prediction(image_path, expected_type):
    """Test a single image prediction"""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict-cnn", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
        
        result = response.json()
        predicted = result['tumor_type']
        confidence = result['confidence']
        
        is_correct = predicted.lower() == expected_type.lower()
        
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"{status} | Expected: {expected_type:12} | Predicted: {predicted:12} | Confidence: {confidence:6.2f}%")
        
        return is_correct
    except Exception as e:
        print(f"❌ Error testing {image_path}: {str(e)}")
        return False

def main():
    print("\n" + "="*80)
    print("BRAIN TUMOR PREDICTION TEST SUITE")
    print("="*80)
    
    results = {
        "glioma": [],
        "meningioma": [],
        "notumor": [],
        "pituitary": []
    }
    
    # Test each tumor type
    for tumor_type in results.keys():
        type_dir = os.path.join(TEST_DIR, tumor_type)
        
        if not os.path.exists(type_dir):
            print(f"\n⚠️  Directory not found: {type_dir}")
            continue
        
        images = list(Path(type_dir).glob("*.jpg"))[:3]  # Test first 3 images of each type
        
        print(f"\n{'─'*80}")
        print(f"Testing {tumor_type.upper()} images ({len(images)} samples):")
        print(f"{'─'*80}")
        
        for img_path in images:
            result = test_prediction(str(img_path), tumor_type)
            results[tumor_type].append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total_correct = 0
    total_tests = 0
    
    for tumor_type, test_results in results.items():
        if test_results:
            correct = sum(test_results)
            total = len(test_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            total_correct += correct
            total_tests += total
            print(f"{tumor_type.upper():15} | {correct}/{total} correct ({accuracy:6.1f}%)")
    
    print(f"{'─'*80}")
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    print(f"{'OVERALL':15} | {total_correct}/{total_tests} correct ({overall_accuracy:6.1f}%)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
