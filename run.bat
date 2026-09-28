@echo off
title Crowd Detection & Counting System
echo ==========================================
echo   Crowd Detection & Counting System
echo ==========================================
echo.
if not exist .venv (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 was not found.
        echo Install Python 3.11 and run this file again.
        pause
        exit /b 1
    )
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Starting Streamlit...
streamlit run app.py
pause
