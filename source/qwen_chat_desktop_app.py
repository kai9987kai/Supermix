import argparse
import atexit
import base64
import html
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import webview
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    webview = None


DEFAULT_BASE_MODEL = (
    r"C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)
APP_ICON_FILENAME = "supermix_qwen_icon.ico"
SPLASH_IMAGE_FILENAME = "supermix_qwen_splash.png"
APP_STATE_DIRNAME = "SupermixQwenDesktop"

LOADING_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Qwen Desktop</title>
  <style>
    :root {
      --bg: #08111e;
      --panel: #111c30;
      --border: #24324c;
      --text: #e6eefc;
      --muted: #9cb1d4;
      --accent: #4f8cff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top, rgba(79, 140, 255, 0.18), transparent 38%),
        linear-gradient(180deg, #0a1424, var(--bg));
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
    }
    .card {
      width: min(700px, calc(100vw - 40px));
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(17, 28, 48, 0.94);
      padding: 28px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 18px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 12px rgba(79, 140, 255, 0.8);
      animation: pulse 1.2s infinite ease-in-out;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 30px;
      line-height: 1.15;
    }
    p {
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }
    .box {
      margin-top: 18px;
      padding: 16px;
      border-radius: 12px;
      background: #0b1322;
      border: 1px solid var(--border);
      font-family: Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      color: #bfd0f0;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(0.82); opacity: 0.55; }
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge"><span class="dot"></span>Supermix Qwen Desktop</div>
    <h1>Starting chat interface</h1>
    <p>The desktop app is launching the local Python chat server and will switch into the live interface as soon as the server reports ready.</p>
    <div class="box">__STATUS__</div>
  </div>
</body>
</html>
"""

ERROR_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Qwen Desktop Error</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #120f16;
      color: #f3e9ef;
      font-family: "Segoe UI", Arial, sans-serif;
    }
    .card {
      width: min(820px, calc(100vw - 40px));
      border: 1px solid #5a2a3b;
      border-radius: 18px;
      background: #1c1420;
      padding: 28px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.4);
    }
    h1 {
      margin: 0 0 12px;
      font-size: 28px;
    }
    p {
      margin: 0 0 14px;
      color: #d9bac7;
      line-height: 1.55;
    }
    .box {
      margin-top: 18px;
      padding: 16px;
      border-radius: 12px;
      background: #100c12;
      border: 1px solid #5a2a3b;
      font-family: Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      color: #ffd5e1;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Startup failed</h1>
    <p>__MESSAGE__</p>
    <div class="box">__DETAILS__</div>
  </div>
</body>
</html>
"""


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def resolve_runtime_state_dir() -> Path:
    override = str(os.environ.get("SUPERMIX_QWEN_DESKTOP_STATE_DIR") or "").strip()
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
    raise PermissionError("Could not create a writable runtime state directory for Supermix Qwen Desktop.")


def render_loading_html(status: str) -> str:
    return LOADING_HTML.replace("__STATUS__", html.escape(status))


def render_error_html(message: str, details: str) -> str:
    return (
        ERROR_HTML
        .replace("__MESSAGE__", html.escape(message))
        .replace("__DETAILS__", html.escape(details))
    )


def render_splash_html(status: str, splash_data_uri: str) -> str:
    image_markup = (
        f'<img class="hero" src="{splash_data_uri}" alt="Supermix Qwen">'
        if splash_data_uri
        else '<div class="hero hero-fallback">SQ</div>'
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Supermix Qwen Desktop</title>
  <style>
    :root {{
      --bg-top: #07101d;
      --bg-bottom: #0b2035;
      --panel: rgba(10, 20, 35, 0.86);
      --border: rgba(124, 164, 215, 0.24);
      --text: #edf4ff;
      --muted: #a8bedf;
      --accent: #63b4ff;
      --warm: #ff9e61;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      overflow: hidden;
      background:
        radial-gradient(circle at 16% 18%, rgba(77, 166, 255, 0.28), transparent 28%),
        radial-gradient(circle at 85% 82%, rgba(255, 152, 88, 0.24), transparent 30%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    .frame {{
      width: min(940px, calc(100vw - 32px));
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: 0 36px 120px rgba(0, 0, 0, 0.45);
      backdrop-filter: blur(12px);
    }}
    .hero {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .hero-fallback {{
      height: 260px;
      display: grid;
      place-items: center;
      font-size: 92px;
      font-weight: 800;
      letter-spacing: 0.06em;
      background:
        radial-gradient(circle at top left, rgba(99, 180, 255, 0.35), transparent 30%),
        linear-gradient(180deg, #0b1628, #102946);
    }}
    .meta {{
      display: grid;
      gap: 14px;
      padding: 22px 24px 26px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }}
    .status {{
      padding: 16px 18px;
      border-radius: 14px;
      background: rgba(7, 14, 24, 0.92);
      border: 1px solid rgba(99, 180, 255, 0.18);
      color: #d4e4fb;
      white-space: pre-wrap;
      font-family: Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
    }}
    .progress {{
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--warm);
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
    }}
    .pulse {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: var(--warm);
      box-shadow: 0 0 18px rgba(255, 158, 97, 0.8);
      animation: pulse 1.1s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(0.7); opacity: 0.45; }}
    }}
  </style>
</head>
<body>
  <div class="frame">
    {image_markup}
    <div class="meta">
      <div class="eyebrow">Local model startup</div>
      <h1>Preparing the Supermix Qwen desktop chat</h1>
      <p>The launcher is starting the local Python model server and will switch into the live chat window as soon as the adapter reports ready.</p>
      <div class="progress"><span class="pulse"></span>Loading model + adapter</div>
      <div class="status">{html.escape(status)}</div>
    </div>
  </div>
</body>
</html>
"""


def iter_search_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add_path(path: Optional[Path]) -> None:
        if path is None:
            return
        for candidate in (path, *path.parents):
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                roots.append(candidate.resolve())

    add_path(Path.cwd())
    add_path(Path(sys.executable).resolve().parent)
    add_path(Path(__file__).resolve().parent)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        add_path(Path(meipass))
    return roots


def iter_runtime_roots(project_root: Optional[Path] = None) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add_path(path: Optional[Path]) -> None:
        if path is None:
            return
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        add_path(Path(meipass))
    add_path(project_root)
    add_path(Path(sys.executable).resolve().parent)
    add_path(Path.cwd())
    add_path(Path(__file__).resolve().parents[1])
    return roots


def resolve_runtime_path(project_root: Path, relative_path: Path, required: bool = False) -> Optional[Path]:
    for root in iter_runtime_roots(project_root):
        candidate = (root / relative_path).resolve()
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Could not resolve runtime path for {relative_path}")
    return None


def resolve_server_script_path(project_root: Path) -> Path:
    return resolve_runtime_path(project_root, Path("source") / "qwen_chat_web_app.py", required=True)


def resolve_asset_path(project_root: Path, asset_name: str) -> Optional[Path]:
    return resolve_runtime_path(project_root, Path("assets") / asset_name, required=False)


def encode_image_as_data_uri(image_path: Optional[Path]) -> str:
    if image_path is None or not image_path.exists():
        return ""
    if image_path.suffix.lower() == ".png":
        mime = "image/png"
    elif image_path.suffix.lower() == ".ico":
        mime = "image/x-icon"
    else:
        mime = "application/octet-stream"
    raw = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{raw}"


def find_project_root() -> Path:
    for root in iter_search_roots():
        has_artifacts = (root / "artifacts").exists()
        has_source = (root / "source" / "qwen_chat_web_app.py").exists()
        if has_artifacts and has_source:
            return root
    return Path.cwd().resolve()


def find_bundled_adapter_dir() -> Optional[Path]:
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    candidate = Path(meipass) / "bundled_latest_adapter"
    if (candidate / "adapter_config.json").exists():
        return candidate.resolve()
    return None


def score_adapter_dir(adapter_dir: Path) -> tuple[int, float]:
    parent = adapter_dir.parent
    benchmark_file = parent / "benchmark_results.json"
    if benchmark_file.exists():
        return (2, benchmark_file.stat().st_mtime)
    checkpoint_meta = parent / "checkpoint_meta.json"
    if checkpoint_meta.exists():
        return (1, checkpoint_meta.stat().st_mtime)
    weight_file = adapter_dir / "adapter_model.safetensors"
    if weight_file.exists():
        return (0, weight_file.stat().st_mtime)
    return (0, adapter_dir.stat().st_mtime)


def find_latest_adapter_dir(project_root: Path) -> Path:
    bundled = find_bundled_adapter_dir()
    if bundled is not None:
        return bundled

    candidates: dict[str, Path] = {}
    patterns = [
        "artifacts/qwen_supermix_enhanced_*/adapter",
        "artifacts/*/adapter",
        "artifacts/**/adapter",
    ]
    for pattern in patterns:
        for candidate in project_root.glob(pattern):
            cfg = candidate / "adapter_config.json"
            weights = candidate / "adapter_model.safetensors"
            if cfg.exists() and weights.exists():
                candidates[str(candidate.resolve())] = candidate.resolve()
    if not candidates:
        raise FileNotFoundError("Could not find any Qwen adapter directory under artifacts.")
    return max(candidates.values(), key=score_adapter_dir)


def resolve_adapter_dir(project_root: Path, explicit_adapter_dir: str) -> Path:
    raw = str(explicit_adapter_dir or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        if not (candidate / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter directory is missing adapter_config.json: {candidate}")
        return candidate
    return find_latest_adapter_dir(project_root)


def resolve_base_model_path(adapter_dir: Path, explicit_base_model: str) -> str:
    if str(explicit_base_model or "").strip():
        return str(explicit_base_model).strip()
    cfg_path = adapter_dir / "adapter_config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_model = str(data.get("base_model_name_or_path") or "").strip()
        if base_model:
            return base_model
    except Exception as exc:
        logging.warning("Failed to read base model path from %s: %s", cfg_path, exc)
    return DEFAULT_BASE_MODEL


def choose_port(host: str, preferred_port: int) -> int:
    if int(preferred_port) > 0:
        return int(preferred_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def resolve_python_executable(explicit_python: str) -> str:
    if str(explicit_python or "").strip():
        return str(explicit_python).strip()
    candidates: list[str] = []
    if not getattr(sys, "frozen", False):
        candidates.append(sys.executable)
    for name in ("python", "python3"):
        path = shutil.which(name)
        if path:
            candidates.append(path)
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not find a usable Python executable. Pass --python_exe explicitly.")


def wait_for_server(url: str, timeout_sec: float, process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.time() + float(timeout_sec)
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:
                log_tail = ""
            raise RuntimeError(
                f"Server process exited early with code {process.returncode}.\n\n{log_tail}"
            )
        try:
            with urllib.request.urlopen(f"{url}/api/status", timeout=2.0) as resp:
                if int(getattr(resp, "status", 200)) == 200:
                    return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for local server at {url}. Last error: {last_error}")


class ServerProcess:
    def __init__(
        self,
        project_root: Path,
        python_exe: str,
        script_path: Path,
        base_model: str,
        adapter_dir: Path,
        host: str,
        port: int,
        device: str,
        log_path: Path,
    ) -> None:
        self.project_root = project_root
        self.python_exe = python_exe
        self.script_path = script_path
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.host = host
        self.port = int(port)
        self.device = device
        self.log_path = log_path
        self.process: Optional[subprocess.Popen[bytes]] = None
        self._log_handle = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [
            self.python_exe,
            "-u",
            str(self.script_path),
            "--base_model",
            self.base_model,
            "--adapter_dir",
            str(self.adapter_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--device",
            self.device,
        ]
        logging.info("Launching server subprocess: %s", cmd)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            logging.info("Stopping server subprocess pid=%s", self.process.pid)
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class DesktopApp:
    def __init__(
        self,
        project_root: Path,
        server_script_path: Path,
        python_exe: str,
        base_model: str,
        adapter_dir: Path,
        host: str,
        port: int,
        device: str,
        server_timeout: float,
        server_log_path: Path,
    ) -> None:
        self.project_root = project_root
        self.server_script_path = server_script_path
        self.python_exe = python_exe
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.host = host
        self.port = int(port)
        self.device = device
        self.server_timeout = float(server_timeout)
        self.server_log_path = server_log_path
        self.server = ServerProcess(
            project_root=project_root,
            python_exe=python_exe,
            script_path=server_script_path,
            base_model=base_model,
            adapter_dir=adapter_dir,
            host=host,
            port=port,
            device=device,
            log_path=server_log_path,
        )

    @property
    def url(self) -> str:
        return self.server.url

    def stop(self) -> None:
        self.server.stop()

    def start_backend(self) -> None:
        self.server.start()
        assert self.server.process is not None
        wait_for_server(
            url=self.url,
            timeout_sec=self.server_timeout,
            process=self.server.process,
            log_path=self.server_log_path,
        )

    def bootstrap(self, window, splash_window=None) -> None:
        try:
            self.start_backend()
            logging.info("Desktop app ready at %s", self.url)
            window.load_url(self.url)
            window.show()
            if splash_window is not None:
                splash_window.destroy()
        except Exception as exc:
            logging.exception("Desktop startup failed")
            details = (
                f"{exc}\n\n"
                f"project_root = {self.project_root}\n"
                f"server_script = {self.server_script_path}\n"
                f"python_exe = {self.python_exe}\n"
                f"adapter_dir = {self.adapter_dir}\n"
                f"base_model = {self.base_model}\n"
                f"device = {self.device}\n"
                f"url = {self.url}\n"
                f"server_log = {self.server_log_path}\n"
            )
            if splash_window is not None:
                try:
                    splash_window.destroy()
                except Exception:
                    logging.exception("Failed to close splash window after startup error")
            window.load_html(render_error_html("The Python chat server could not be started.", details))
            window.show()

    def run_server_only(self) -> None:
        self.start_backend()
        logging.info("Server-only mode ready at %s", self.url)
        print(self.url)
        try:
            while self.server.process is not None and self.server.process.poll() is None:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logging.info("Server-only mode interrupted by user")
        finally:
            self.stop()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Desktop launcher for the latest Supermix Qwen chat UI.")
    ap.add_argument("--root", default="", help="Optional project root. Defaults to auto-detect.")
    ap.add_argument("--python_exe", default="", help="Optional Python executable used to run the server script.")
    ap.add_argument("--adapter_dir", default="", help="Optional adapter dir. Defaults to the newest available adapter.")
    ap.add_argument("--base_model", default="", help="Optional base model path. Defaults to adapter_config.json value.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="Port for the embedded local server. 0 picks a free port.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--width", type=int, default=1360)
    ap.add_argument("--height", type=int, default=920)
    ap.add_argument("--title", default="Supermix Qwen Desktop")
    ap.add_argument("--server_timeout", type=float, default=240.0)
    ap.add_argument("--server_only", action="store_true", help="Run the server subprocess without opening a webview.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.root).resolve() if str(args.root or "").strip() else find_project_root()
    state_dir = resolve_runtime_state_dir() if getattr(sys, "frozen", False) else project_root
    log_path = (
        state_dir / f"{Path(sys.executable).stem}.log"
        if getattr(sys, "frozen", False)
        else project_root / "supermix_qwen_desktop.log"
    )
    configure_logging(log_path)

    adapter_dir = resolve_adapter_dir(project_root, args.adapter_dir)
    base_model = resolve_base_model_path(adapter_dir, args.base_model)
    server_script_path = resolve_server_script_path(project_root)
    python_exe = resolve_python_executable(args.python_exe)
    port = choose_port(args.host, args.port)
    server_log_path = (
        state_dir / "supermix_qwen_server.log"
        if getattr(sys, "frozen", False)
        else project_root / "supermix_qwen_server.log"
    )

    app = DesktopApp(
        project_root=project_root,
        server_script_path=server_script_path,
        python_exe=python_exe,
        base_model=base_model,
        adapter_dir=adapter_dir,
        host=args.host,
        port=port,
        device=args.device,
        server_timeout=args.server_timeout,
        server_log_path=server_log_path,
    )
    atexit.register(app.stop)

    if args.server_only:
        app.run_server_only()
        return

    if webview is None:
        raise RuntimeError("pywebview is not installed. Run `python -m pip install pywebview` before launching the desktop app.")

    status_lines = [
        f"project_root = {project_root}",
        f"state_dir = {state_dir}",
        f"server_script = {server_script_path}",
        f"python_exe = {python_exe}",
        f"adapter_dir = {adapter_dir}",
        f"base_model = {base_model}",
        f"device = {args.device}",
        f"target_url = {app.url}",
    ]
    icon_path = resolve_asset_path(project_root, APP_ICON_FILENAME)
    splash_path = resolve_asset_path(project_root, SPLASH_IMAGE_FILENAME)
    splash_data_uri = encode_image_as_data_uri(splash_path)
    show_startup_splash = bool(splash_data_uri)

    window = webview.create_window(
        args.title,
        html=render_loading_html("\n".join(status_lines)),
        width=int(args.width),
        height=int(args.height),
        min_size=(980, 720),
        text_select=True,
        confirm_close=True,
        background_color="#08111e",
        hidden=show_startup_splash,
    )
    splash_window = None
    if show_startup_splash:
        splash_window = webview.create_window(
            f"{args.title} Loading",
            html=render_splash_html("\n".join(status_lines), splash_data_uri),
            width=960,
            height=560,
            min_size=(960, 560),
            resizable=False,
            frameless=True,
            easy_drag=True,
            shadow=True,
            on_top=True,
            background_color="#07101d",
        )
    webview.start(
        app.bootstrap,
        args=(window, splash_window),
        debug=False,
        http_server=False,
        icon=str(icon_path) if icon_path else None,
    )


if __name__ == "__main__":
    main()
