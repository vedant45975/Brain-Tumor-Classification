"""Test API endpoint with one image"""
import requests
import os
from pathlib import Path

# Pick a test image
test_img = r"C:\Users\pramo\Downloads\archive (1)\BrainTumor_1\Test\glioma\Te-glTr_0000.jpg"

print(f"Testing API endpoint with: {Path(test_img).name}")
print("="*60)

try:
    with open(test_img, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/predict-cnn', files=files, timeout=5)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to API at localhost:8000")
    print("Make sure the FastAPI server is running!")
except Exception as e:
    print(f"ERROR: {e}")
