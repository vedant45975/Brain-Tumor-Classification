@echo off
cd /d "C:\Users\pramo\Downloads\Brain_Tumor_Classification_Using_Quantum-main\Brain_Tumor_Classification_Using_Quantum-main"
call .venv\Scripts\activate.bat
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
pause
