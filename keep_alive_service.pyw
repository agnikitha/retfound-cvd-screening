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

def check_and_extract_tunnel_url():
    if os.path.exists(TUNNEL_LOG):
        try:
            with open(TUNNEL_LOG, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if matches:
                    latest_url = matches[-1]
                    with open(URL_FILE, "w", encoding="utf-8") as uf:
                        uf.write(latest_url + "\n")
        except Exception:
            pass

def run_forever():
    os.chdir(BASE_DIR)
    python_exe = sys.executable

    while True:
        # 1. Maintain local unified_web_gui.py
        if not is_port_open(5050):
            with open(LOG_PATH, "a", encoding="utf-8") as logf:
                logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting unified_web_gui.py on port 5050...\n")
            subprocess.Popen(
                [python_exe, SCRIPT_PATH],
                cwd=BASE_DIR,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            time.sleep(5)

        # 2. Maintain Cloudflare Tunnel
        if is_port_open(5050) and os.path.exists(CLOUDFLARED_EXE) and not is_process_running("cloudflared.exe"):
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
