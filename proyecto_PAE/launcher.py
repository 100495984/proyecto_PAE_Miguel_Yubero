import subprocess
import sys
import os

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "code", "app.py")

    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--browser.serverAddress=localhost"
    ], check=True)

if __name__ == "__main__":
    main()