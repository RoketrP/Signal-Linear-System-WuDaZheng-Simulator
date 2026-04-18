import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from streamlit import config as st_config
from streamlit.web import bootstrap


APP_NAME = "Signal-Linear-System-WuDazheng"
HOST = os.environ.get("SLS_HOST", "127.0.0.1")
DEFAULT_PORT = 8501


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port() -> int:
    forced_port = os.environ.get("SLS_PORT")
    if forced_port:
        return int(forced_port)

    for port in range(DEFAULT_PORT, DEFAULT_PORT + 50):
        if port_available(HOST, port):
            return port
    raise RuntimeError("未找到可用端口，请关闭占用的 Streamlit 进程后重试。")


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
    port = resolve_port()

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", HOST)
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)

    st_config.set_option("server.headless", True)
    st_config.set_option("server.address", HOST)
    st_config.set_option("server.port", port)
    st_config.set_option("browser.serverAddress", HOST)
    st_config.set_option("browser.gatherUsageStats", False)
    st_config.set_option("global.developmentMode", False)

    global PORT
    PORT = port
    threading.Thread(target=open_browser_later, daemon=True).start()

    bootstrap.run(
        str(main_script_path()),
        False,
        [],
        {
            "server.headless": True,
            "server.address": HOST,
            "server.port": port,
            "browser.serverAddress": HOST,
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
        },
    )


if __name__ == "__main__":
    main()
