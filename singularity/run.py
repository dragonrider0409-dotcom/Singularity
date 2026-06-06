"""
Launch Singularity as a desktop window.

Starts the local server in the background, then opens a native window
(pywebview). If no webview backend is available it falls back to the
default browser so it always works.
"""
import threading
import time

import uvicorn
from backend.main import app

URL = "http://127.0.0.1:8000"


def _serve():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def main():
    threading.Thread(target=_serve, daemon=True).start()
    time.sleep(1.5)
    try:
        import webview  # pywebview
        webview.create_window("Singularity", URL, width=1200, height=880,
                              min_size=(940, 660))
        webview.start()
    except Exception:
        import webbrowser
        webbrowser.open(URL)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
