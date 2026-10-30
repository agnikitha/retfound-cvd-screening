@echo off
title RETFound-CVD Clinical AI Screening Platform
color 0B
cd /d "D:\new pro"

echo ==============================================================================
echo             RETFound-CVD CLINICAL AI SCREENING PLATFORM
echo ==============================================================================
echo.
echo  [1] Open Online Streamlit Cloud GUI (agnikitha/retfound-cvd-screening)
echo  [2] Run Local Streamlit Medical White GUI (http://localhost:8501)
echo  [3] Run Local Clinical Portal (http://127.0.0.1:5050)
echo.
echo ==============================================================================
set /p choice="Enter choice [1, 2, or 3] (Default is 1): "

if "%choice%"=="2" goto LOCAL_STREAMLIT
if "%choice%"=="3" goto LOCAL_FLASK

:STREAMLIT_CLOUD
echo.
echo Opening Streamlit Cloud GUI for agnikitha/retfound-cvd-screening...
start https://share.streamlit.io/agnikitha/retfound-cvd-screening/main/streamlit_app.py
goto END

:LOCAL_STREAMLIT
echo.
echo Starting Local Streamlit White Theme GUI on http://localhost:8501...
start "" py -3.13 -m streamlit run streamlit_app.py --server.port 8501
timeout /t 3 >nul
start http://localhost:8501
goto END

:LOCAL_FLASK
echo.
echo Starting Local Clinical Portal on http://127.0.0.1:5050...
start "" py -3.13 unified_web_gui.py
timeout /t 4 >nul
start http://127.0.0.1:5050
goto END

:END
echo.
echo GUI launcher executed!
timeout /t 3 >nul
