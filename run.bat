@echo off
REM BFS-LLM Execution Script for Windows
REM This script runs the concept tree expansion system

echo ========================================
echo BFS-LLM: Concept Tree Expansion
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies if needed
echo Checking dependencies...
pip install -q -r requirements.txt
echo.

REM Check if OPENAI_API_KEY is set
if "%OPENAI_API_KEY%"=="" (
    echo WARNING: OPENAI_API_KEY environment variable not set
    echo Please set it with: set OPENAI_API_KEY=your-key-here
    echo Or create a .env file based on .env.example
    echo.
    echo Press any key to continue anyway...
    pause > nul
)

REM Run the main script
python main.py

echo.
echo ========================================
echo Execution finished
echo ========================================
pause
