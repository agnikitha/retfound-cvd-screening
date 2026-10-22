@echo off
setlocal enabledelayedexpansion
title RETFound Live Public Portal
cd /d "D:\new pro"

if exist "CLOUDFLARE_TUNNEL_URL.txt" (
    set /p TUNNEL_URL=<"CLOUDFLARE_TUNNEL_URL.txt"
    echo ==============================================================
    echo  RETFound Multi-Disease & Cardiovascular AI Screening Portal
    echo ==============================================================
    echo.
    echo  Local Address:  http://127.0.0.1:5050
    echo  Public Address: !TUNNEL_URL!
    echo.
    echo  Opening in your default browser...
    start "" "!TUNNEL_URL!"
) else (
    echo [!] Tunnel URL not generated yet. Opening local address...
    start "" "http://127.0.0.1:5050"
)
pause
