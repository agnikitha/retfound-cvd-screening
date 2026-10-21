import os
import sys
import time
import socket
import subprocess
import urllib.request

BASE_DIR = r"D:\new pro"

def is_port_open(port=5050):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(('127.0.0.1', port)) == 0

def launch():
    print("==============================================================================")
    print("          RETFound-CVD PERMANENT BACKGROUND SERVICE & LAUNCHER                ")
    print("==============================================================================")
    os.chdir(BASE_DIR)

    pyw = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pyw):
        pyw = sys.executable

    # 1. Install into Windows Startup folder so it survives PC reboots permanently
    if sys.platform == "win32" and "APPDATA" in os.environ:
        startup_dir = os.path.join(os.environ["APPDATA"], r"Microsoft\Windows\Start Menu\Programs\Startup")
        if os.path.exists(startup_dir):
            vbs_path = os.path.join(startup_dir, "RETFound_CVD_AutoStart.vbs")
            vbs_code = (
                'Set WshShell = CreateObject("WScript.Shell")\n'
                f'WshShell.Run """{pyw}"" ""D:\\new pro\\keep_alive_service.pyw""", 0, False\n'
            )
            with open(vbs_path, "w", encoding="utf-8") as f:
                f.write(vbs_code)
            print(f"[1/3] Registered permanent Windows Startup service: {vbs_path}")

    # 2. Launch detached background watchdog if port 5050 is not yet active
    if not is_port_open(5050):
        print("[2/3] Starting detached permanent watchdog (keep_alive_service.pyw)...")
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [pyw, os.path.join(BASE_DIR, "keep_alive_service.pyw")],
            cwd=BASE_DIR,
            creationflags=flags
        )
    else:
        print("[2/3] Port 5050 is already active and listening!")

    # 3. Wait until http://127.0.0.1:5050 responds with HTTP 200
    print("[3/3] Waiting for http://127.0.0.1:5050 to respond...")
    for i in range(45):
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:5050", timeout=2)
            if resp.getcode() == 200:
                print(f" -> Server is LIVE and verified (HTTP 200 OK after {i+1}s)!")
                break
        except Exception:
            time.sleep(1)

    if sys.platform == "win32":
        os.startfile("http://127.0.0.1:5050")

if __name__ == "__main__":
    launch()
