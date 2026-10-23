@echo off
title RETFound-CVD Permanent Clinical Web Launcher
color 0A
cd /d "%~dp0"

echo ==============================================================================
echo          RETFound-CVD CLINICAL SCREENING & EXPLAINABLE AI PLATFORM
echo ==============================================================================
echo.

:: Start the permanent background watchdog service if port 5050 is not already listening
netstat -aon | find ":5050" | find "LISTENING" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/2] Starting permanent background server (keep_alive_service.pyw)...
    start "" py -3.13 "D:\new pro\keep_alive_service.pyw"
    echo       Waiting for AI models to load into memory...
    timeout /t 5 /nobreak >nul
) else (
    echo [1/2] Server is already running on port 5050!
)

echo [2/2] Opening http://127.0.0.1:5050 in your default browser...
start http://127.0.0.1:5050

echo.
echo ==============================================================================
echo  SUCCESS! The GUI is permanently running in the background at:
echo  http://127.0.0.1:5050
echo  Even if you close this window, the web server stays LIVE!
echo ==============================================================================
timeout /t 4 >nul
