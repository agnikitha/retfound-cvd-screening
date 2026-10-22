import os
import sys
import time
import socket
import subprocess
import re

BASE_DIR = r"D:\new pro"
SCRIPT_PATH = os.path.join(BASE_DIR, "unified_web_gui.py")
LOG_PATH = os.path.join(BASE_DIR, "gui_server.log")
CLOUDFLARED_EXE = os.path.join(BASE_DIR, "cloudflared.exe")
TUNNEL_LOG = os.path.join(BASE_DIR, "cloudflared.log")
URL_FILE = os.path.join(BASE_DIR, "CLOUDFLARE_TUNNEL_URL.txt")
DESKTOP_HTML = os.path.expanduser(r"~\Desktop\OPEN_RETFOUND_AI_PORTAL.html")

# Explicit Python 3.13 executable containing PyTorch, Timm, Flask, OpenCV
PY_313 = r"C:\Users\Nikitha AG\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
if not os.path.exists(PY_313):
    PY_313 = sys.executable

def is_port_open(port=5050):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(('127.0.0.1', port)) == 0

def is_process_running(proc_name="cloudflared.exe"):
    try:
        out = subprocess.check_output(f'tasklist /FI "IMAGENAME eq {proc_name}"', shell=True, text=True, stderr=subprocess.DEVNULL)
        return proc_name.lower() in out.lower()
    except Exception:
        return False

def update_tunnel_launchers(url):
    try:
        with open(URL_FILE, "w", encoding="utf-8") as uf:
            uf.write(url + "\n")
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url='{url}'" />
    <title>Redirecting to RETFound Live Screening Portal</title>
</head>
<body style="font-family:sans-serif; text-align:center; padding-top:50px; background:#0f172a; color:#fff;">
    <h2>Opening RETFound Clinical Screening Portal...</h2>
    <p>If not redirected automatically, <a href="{url}" style="color:#38bdf8; font-size:18px;">Click Here to Open</a></p>
</body>
</html>"""
        with open(DESKTOP_HTML, "w", encoding="utf-8") as dh:
            dh.write(html_content)
    except Exception:
        pass

def check_and_extract_tunnel_url():
    if os.path.exists(TUNNEL_LOG):
        try:
            with open(TUNNEL_LOG, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if matches:
                    latest_url = matches[-1]
                    update_tunnel_launchers(latest_url)
        except Exception:
            pass

def run_forever():
    os.chdir(BASE_DIR)

    while True:
        # 1. Maintain local unified_web_gui.py with Python 3.13
        if not is_port_open(5050):
            with open(LOG_PATH, "a", encoding="utf-8") as logf:
                logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting unified_web_gui.py on port 5050 via Python 3.13...\n")
            subprocess.Popen(
                [PY_313, SCRIPT_PATH],
                cwd=BASE_DIR,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            time.sleep(5)

        # 2. Maintain Cloudflare Tunnel
        if is_port_open(5050) and os.path.exists(CLOUDFLARED_EXE):
            # Check if tunnel is disconnected or in error
            tunnel_dead = False
            if os.path.exists(TUNNEL_LOG):
                try:
                    with open(TUNNEL_LOG, "r", encoding="utf-8", errors="ignore") as tf:
                        tail = tf.read()[-1000:]
                        if "Unauthorized: Tunnel not found" in tail:
                            tunnel_dead = True
                except Exception:
                    pass

            if tunnel_dead:
                try:
                    subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

            if not is_process_running("cloudflared.exe"):
                tunnel_log_fp = open(TUNNEL_LOG, "a", encoding="utf-8")
                tunnel_log_fp.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Cloudflare Tunnel...\n")
                tunnel_log_fp.flush()
                subprocess.Popen(
                    [CLOUDFLARED_EXE, "tunnel", "--url", "http://127.0.0.1:5050"],
                    cwd=BASE_DIR,
                    stdout=tunnel_log_fp,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                time.sleep(4)
                check_and_extract_tunnel_url()

        time.sleep(5)

if __name__ == "__main__":
    run_forever()
