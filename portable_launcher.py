import os
import sys
import time
import webbrowser
import subprocess

def launch():
    print("==============================================================================")
    print("          RETFound-CVD CLINICAL SCREENING & EXPLAINABLE AI PLATFORM           ")
    print("==============================================================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 1. Check port 5050 and kill stale processes on Windows
    if sys.platform == 'win32':
        try:
            output = subprocess.check_output('netstat -aon | find ":5050" | find "LISTENING"', shell=True).decode()
            for line in output.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    print(f"Releasing occupied port 5050 (PID: {pid})...")
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
        except Exception:
            pass

    # 2. Launch unified_web_gui.py
    print("Starting Flask web server on http://127.0.0.1:5050 ...")
    env = os.environ.copy()
    server_process = subprocess.Popen([sys.executable, "unified_web_gui.py"], cwd=script_dir, env=env)
    
    # 3. Wait and open browser
    time.sleep(2.5)
    print("Opening web dashboard in your default browser...")
    webbrowser.open("http://127.0.0.1:5050")
    
    print("\nServer is active! Press Ctrl+C in this terminal to shut down.")
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nStopping RETFound-CVD platform...")
        server_process.terminate()

if __name__ == '__main__':
    launch()
