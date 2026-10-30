@echo off
title RETFound-CVD Offline Clinical AI Portal
cd /d "%~dp0"
echo ==============================================================================
echo             RETFound-CVD CLINICAL AI SCREENING PLATFORM
echo                     (Standalone Offline Edition)
echo ==============================================================================
echo.
echo Launching RETFound Clinical Portal offline in default web browser...
start "" "retfound_clinical_portal.html"
echo.
echo Done! Portal opened with zero cloud dependencies and medical white background.
timeout /t 3 >nul
