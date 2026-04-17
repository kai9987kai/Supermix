import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch qwen_chat_web_app.py as a detached process.")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out_log", required=True)
    ap.add_argument("--err_log", required=True)
    ap.add_argument("--pid_file", required=True)
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    out_log = Path(args.out_log)
    err_log = Path(args.err_log)
    pid_file = Path(args.pid_file)

    out_log.parent.mkdir(parents=True, exist_ok=True)
    err_log.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    creationflags = 0
    creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    with out_log.open("w", encoding="utf-8") as stdout, err_log.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(root_dir / "source" / "qwen_chat_web_app.py"),
                "--base_model",
                str(args.base_model),
                "--adapter_dir",
                str(args.adapter_dir),
                "--host",
                str(args.host),
                "--port",
                str(args.port),
            ],
            cwd=str(root_dir),
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            close_fds=True,
        )

    pid_file.write_text(str(proc.pid), encoding="utf-8")
    print(proc.pid)


if __name__ == "__main__":
    main()
