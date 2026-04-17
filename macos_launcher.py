import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

from streamlit import config as st_config
from streamlit.web import bootstrap


APP_NAME = "Signal-Linear-System-WuDazheng"
HOST = os.environ.get("SLS_HOST", "127.0.0.1")
PORT = int(os.environ.get("SLS_PORT", "8501"))


def resource_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def main_script_path() -> Path:
    candidates = [
        resource_root() / "app.py",
        Path(__file__).resolve().parent / "app.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到打包后的 app.py 主程序文件。")


def open_browser_later() -> None:
    url = f"http://{HOST}:{PORT}"
    time.sleep(2.0)
    webbrowser.open(url)


def main() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", HOST)
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(PORT))

    st_config.set_option("server.headless", True)
    st_config.set_option("server.address", HOST)
    st_config.set_option("server.port", PORT)
    st_config.set_option("browser.serverAddress", HOST)
    st_config.set_option("browser.gatherUsageStats", False)
    st_config.set_option("global.developmentMode", False)

    threading.Thread(target=open_browser_later, daemon=True).start()

    bootstrap.run(
        str(main_script_path()),
        False,
        [],
        {
            "server.headless": True,
            "server.address": HOST,
            "server.port": PORT,
            "browser.serverAddress": HOST,
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
        },
    )


if __name__ == "__main__":
    main()
