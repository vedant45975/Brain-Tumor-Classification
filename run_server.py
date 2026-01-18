#!/usr/bin/env python
"""Simple server runner for the FastAPI app"""
import sys
import os

# Add backend to path
sys.path.insert(0, '/backend')

os.chdir(r'C:\Users\pramo\Downloads\Brain_Tumor_Classification_Using_Quantum-main\Brain_Tumor_Classification_Using_Quantum-main\backend')

from app.main import app
from uvicorn import run

if __name__ == "__main__":
    run(app, host="127.0.0.1", port=8000, log_level="warning")
