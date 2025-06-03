import sys
import os
import subprocess
from flask import Flask, send_from_directory
from threading import Thread

WORKSPACE_ROOT = "/data/disk0/Workspace/Compiler-Toolchain/Compiler-Toolchain"
sys.path.insert(0, WORKSPACE_ROOT)
os.chdir(WORKSPACE_ROOT)

app = Flask(__name__)

STATIC_FOLDER = os.path.join(WORKSPACE_ROOT, "CT", "WebUI", "dist")
API_PATH = os.path.join(WORKSPACE_ROOT, "CT", "WebUI", "api_test.py")

@app.route('/')
def index():
    return send_from_directory(STATIC_FOLDER, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(STATIC_FOLDER, path)

def run_frontend():
    app.run(host='0.0.0.0', port=6348)

def run_backend():
    backend_script = API_PATH
    subprocess.run([sys.executable, backend_script])

if __name__ == '__main__':
    frontend_thread = Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    run_backend()