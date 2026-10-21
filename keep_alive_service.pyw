import os
import sys
import time
import socket
import subprocess

BASE_DIR = r"D:\new pro"
SCRIPT_PATH = os.path.join(BASE_DIR, "unified_web_gui.py")
LOG_PATH = os.path.join(BASE_DIR, "gui_server.log")

def is_port_open(port=5050):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(('127.0.0.1', port)) == 0

def run_forever():
    os.chdir(BASE_DIR)
    python_exe = sys.executable
    # Prefer pythonw.exe if available so no console window pops up
    pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw_exe):
        pythonw_exe = python_exe

    while True:
        if not is_port_open(5050):
            with open(LOG_PATH, "a", encoding="utf-8") as logf:
                logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting unified_web_gui.py on port 5050...\n")
                proc = subprocess.Popen(
                    [python_exe, SCRIPT_PATH],
                    cwd=BASE_DIR,
                    stdout=logf,
                    stderr=logf,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                proc.wait()
        time.sleep(3)

if __name__ == "__main__":
    run_forever()
