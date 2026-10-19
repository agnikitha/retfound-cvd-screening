@echo off
title RETFound-CVD Clinical Screening & Explainable AI Platform
color 0A
cls
echo ==============================================================================
echo          RETFound-CVD CLINICAL SCREENING & EXPLAINABLE AI PLATFORM
echo ==============================================================================
echo.
echo [1/3] Checking environment and port 5050 availability...

:: Kill any lingering processes on port 5050
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5050" ^| find "LISTENING"') do (
    echo  -> Releasing occupied port 5050 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/3] Launching RETFound-CVD Unified Web GUI...
cd /d "%~dp0"

:: Start the python Flask server in background or foreground
start "RETFound CVD Server" /b python "unified_web_gui.py"

:: Wait 3 seconds for server initialization
timeout /t 3 /nobreak >nul

echo [3/3] Opening Clinical Dashboard in your default browser...
start http://127.0.0.1:5050

echo.
echo ==============================================================================
echo  Platform is now LIVE and PERMANENTLY AVAILABLE at: http://127.0.0.1:5050
echo ==============================================================================
echo  - Upload any fundus image (PNG/JPG)
echo  - View 5-Disease probabilities, CIMT thickness (mm), and CVD risk score
echo  - Interactive Layer 20 Grad-CAM microvascular heatmap with genuine reports
echo.
echo  Keep this terminal window open to keep the service running.
echo  Press Ctrl+C to stop the platform.
echo ==============================================================================
echo.

:: Keep script alive to monitor python process
python "unified_web_gui.py"
pause
