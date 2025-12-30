@echo off
REM Launch the Trading Bot Dashboard

echo ================================================================
echo          ToS Trading Bot - Dashboard Launcher
echo ================================================================
echo.

REM Check for required packages
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install fastapi uvicorn websockets
)

echo Starting dashboard server...
echo.
echo Dashboard: http://localhost:8000
echo API Docs:  http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

cd dashboard\backend
python server.py
