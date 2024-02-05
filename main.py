import sys
import subprocess
import os
import time
import streamlit as st





def launch_streamlit_app(app_file):
    cmd = [sys.executable, "-m", "streamlit", "run", app_file]
    return subprocess.Popen(cmd, start_new_session=True)


if __name__ == "__main__":
    os.makedirs("ApiHelpers/logs", exist_ok=True)
    streamlit_process = launch_streamlit_app("NLP_model_site.py")
    while True:
        time.sleep(5)