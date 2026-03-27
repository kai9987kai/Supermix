from __future__ import annotations

import argparse
import atexit
import base64
import html
import logging
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from werkzeug.serving import make_server

from multimodel_catalog import DEFAULT_MODELS_DIR, discover_model_records
from qwen_chat_desktop_app import (
    APP_ICON_FILENAME,
    SPLASH_IMAGE_FILENAME,
    configure_logging,
    encode_image_as_data_uri,
    find_bundled_base_model_dir,
    iter_runtime_roots,
    resolve_asset_path,
)
from supermix_multimodel_web_app import build_app
from multimodel_runtime import UnifiedModelManager

try:
    import webview
except ImportError:  # pragma: no cover
    webview = None


APP_STATE_DIRNAME = "SupermixStudioDesktop"

LOADING_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supermix Studio</title>
<style>
  body{margin:0;min-height:100vh;display:grid;place-items:center;background:
    radial-gradient(circle at 12% 18%, rgba(112,184,255,.22), transparent 26%),
    radial-gradient(circle at 86% 84%, rgba(99,217,200,.16), transparent 28%),
    linear-gradient(180deg,#05101b,#091624);color:#eef5ff;font-family:"Aptos","Bahnschrift","Segoe UI",sans-serif}
  .card{width:min(760px,calc(100vw - 32px));padding:28px;border-radius:26px;border:1px solid rgba(135,164,203,.18);
    background:rgba(10,20,33,.92);box-shadow:0 30px 80px rgba(0,0,0,.34)}
  .eyebrow{display:inline-flex;align-items:center;gap:10px;color:#70b8ff;font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700}
  .eyebrow::before{content:"";width:10px;height:10px;border-radius:999px;background:#70b8ff;box-shadow:0 0 16px rgba(112,184,255,.8)}
  h1{margin:14px 0 12px;font-size:32px;line-height:1.04;font-family:"Bahnschrift","Segoe UI Semibold",sans-serif}
  p{margin:0;color:#9eb3cf;line-height:1.6}
  .box{margin-top:18px;padding:16px;border-radius:16px;background:#07111b;border:1px solid rgba(135,164,203,.14);white-space:pre-wrap;
    color:#d5e5f9;font-family:Consolas,"Cascadia Code",monospace;font-size:12px;line-height:1.55}
</style></head>
<body><div class="card"><div class="eyebrow">Bundled Local Runtime</div><h1>Starting Supermix Studio</h1><p>The desktop shell is bringing up the embedded multimodel server and will switch into the live interface as soon as it reports ready.</p><div class="box">__STATUS__</div></div></body></html>
"""


def render_loading_html(status: str) -> str:
    return LOADING_HTML.replace("__STATUS__", html.escape(status))


def choose_port(host: str, preferred_port: int) -> int:
    if int(preferred_port) > 0:
        return int(preferred_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def resolve_runtime_path(relative_path: Path) -> Optional[Path]:
    for root in iter_runtime_roots(None):
        candidate = (root / relative_path).resolve()
        if candidate.exists():
            return candidate
    return None


def resolve_models_dir(explicit_models_dir: str) -> Path:
    raw = str(explicit_models_dir or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    bundled = resolve_runtime_path(Path("bundled_models"))
    if bundled is not None:
        return bundled
    return DEFAULT_MODELS_DIR.resolve()


def resolve_runtime_state_dir() -> Path:
    override = str(os.environ.get("SUPERMIX_STUDIO_STATE_DIR") or "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    for env_name in ("LOCALAPPDATA", "APPDATA"):
        raw = str(os.environ.get(env_name) or "").strip()
        if raw:
            candidates.append(Path(raw) / APP_STATE_DIRNAME)
    candidates.append(Path.home() / f".{APP_STATE_DIRNAME}")
    candidates.append(Path(tempfile.gettempdir()) / APP_STATE_DIRNAME)

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return candidate.resolve()
        except Exception:
            continue
    raise PermissionError("Could not create a writable runtime state directory for Supermix Studio.")


class EmbeddedServer:
    def __init__(self, app, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self._server = make_server(self.host, self.port, app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, name="supermix-studio-server", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)


def wait_for_server(url: str, timeout_sec: float) -> None:
    deadline = time.time() + float(timeout_sec)
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/status", timeout=2.0) as resp:
                if int(getattr(resp, "status", 200)) == 200:
                    return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for embedded server at {url}. Last error: {last_error}")


def build_manager(models_dir: Path, state_dir: Path, device_preference: str) -> UnifiedModelManager:
    bundled_base = find_bundled_base_model_dir()
    if bundled_base is not None:
        os.environ["SUPERMIX_QWEN_BASE_MODEL_DIR"] = str(bundled_base)
    records = tuple(discover_model_records(models_dir=models_dir))
    if not records:
        raise RuntimeError(f"No supported model zips were found in {models_dir}")
    return UnifiedModelManager(
        records=records,
        extraction_root=state_dir / "extracted_models",
        generated_dir=state_dir / "generated_images",
        device_preference=device_preference,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desktop launcher for the bundled Supermix Studio multimodel UI.")
    parser.add_argument("--models_dir", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--width", type=int, default=1460)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--title", default="Supermix Studio")
    parser.add_argument("--server_timeout", type=float, default=240.0)
    parser.add_argument("--device_preference", default="cuda,npu,xpu,cpu,dml,mps")
    parser.add_argument("--server_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dir = resolve_runtime_state_dir() if getattr(sys, "frozen", False) else (Path.cwd().resolve() / "run_state_studio")
    log_path = (
        state_dir / f"{Path(sys.executable).stem}.log"
        if getattr(sys, "frozen", False)
        else Path.cwd().resolve() / "supermix_studio_desktop.log"
    )
    configure_logging(log_path)

    models_dir = resolve_models_dir(args.models_dir)
    manager = build_manager(models_dir=models_dir, state_dir=state_dir, device_preference=str(args.device_preference))
    app = build_app(manager)
    port = choose_port(args.host, args.port)
    server = EmbeddedServer(app, args.host, port)
    atexit.register(server.stop)

    status_lines = [
        f"models_dir = {models_dir}",
        f"state_dir = {state_dir}",
        f"models_detected = {len(manager.records)}",
        f"device = {manager.device_info.get('resolved', manager.device)}",
        f"url = {server.url}",
    ]

    server.start()
    wait_for_server(server.url, args.server_timeout)
    logging.info("Supermix Studio ready at %s", server.url)
    print(server.url)

    if args.server_only:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return

    if webview is None:
        raise RuntimeError("pywebview is not installed in this environment.")

    icon_path = resolve_asset_path(Path.cwd().resolve(), APP_ICON_FILENAME)
    splash_path = resolve_asset_path(Path.cwd().resolve(), SPLASH_IMAGE_FILENAME)
    splash_data_uri = encode_image_as_data_uri(splash_path)

    window = webview.create_window(
        args.title,
        url=server.url,
        width=int(args.width),
        height=int(args.height),
        min_size=(1080, 760),
        text_select=True,
        confirm_close=True,
        background_color="#071019",
    )

    if splash_data_uri:
        splash = webview.create_window(
            f"{args.title} Loading",
            html=render_loading_html("\n".join(status_lines)),
            width=900,
            height=520,
            min_size=(900, 520),
            resizable=False,
            frameless=True,
            easy_drag=True,
            shadow=True,
            on_top=True,
            background_color="#071019",
        )

        def close_splash():
            time.sleep(1.0)
            try:
                splash.destroy()
            except Exception:
                logging.exception("Failed to close splash window")

        threading.Thread(target=close_splash, daemon=True).start()

    webview.start(debug=False, http_server=False, icon=str(icon_path) if icon_path else None)


if __name__ == "__main__":
    main()
