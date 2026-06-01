from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import ray


@ray.remote
class NodeProcActor:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.log_path: str | None = None

    def node_ip(self) -> str:
        import ray

        return ray.util.get_node_ip_address()

    def probe_imports(self, python: str, env: dict[str, str]) -> dict[str, object]:
        cmd = [
            python,
            "-c",
            (
                "import importlib.util, json, os;"
                "mods=['accelerate','transformers','triton'];"
                "print(json.dumps({"
                "'python': os.sys.executable,"
                "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),"
                "'mods': {m: bool(importlib.util.find_spec(m)) for m in mods}"
                "}, ensure_ascii=True))"
            ),
        ]
        out = subprocess.run(
            cmd,
            env={**os.environ, **env},
            text=True,
            capture_output=True,
            check=False,
        )
        return {"returncode": out.returncode, "stdout": out.stdout, "stderr": out.stderr}

    def start(self, cmd: str, env: dict[str, str], cwd: str, log_path: str) -> dict[str, object]:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("process already running")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
        merged_env = dict(os.environ)
        merged_env.update(env)
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=cwd,
            env=merged_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.log_path = log_path
        return {"pid": int(self.proc.pid), "log_path": log_path}

    def poll(self) -> dict[str, object]:
        if self.proc is None:
            return {"running": False, "started": False}
        code = self.proc.poll()
        return {
            "running": code is None,
            "started": True,
            "returncode": None if code is None else int(code),
            "pid": int(self.proc.pid),
            "log_path": self.log_path,
        }

    def read_from(self, start: int) -> dict[str, object]:
        if self.log_path is None:
            return {"start": start, "end": start, "text": ""}
        path = Path(self.log_path)
        if not path.exists():
            return {"start": start, "end": start, "text": ""}
        data = path.read_text(encoding="utf-8", errors="replace")
        end = len(data)
        if start >= end:
            return {"start": start, "end": end, "text": ""}
        return {"start": start, "end": end, "text": data[start:end]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ray-address", default="10.11.26.106:6379")
    p.add_argument("--ray-namespace", default="tinker")
    p.add_argument("--node-ip", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--python", default="/usr/bin/python3.12")
    p.add_argument(
        "--quant-script",
        default=str(Path(__file__).resolve().parent / "quantize_glm5_finegrained_fp8.py"),
    )
    p.add_argument("--workdir", default="/vePFS-Mindverse/share/code/tinker-server-aliyun")
    p.add_argument(
        "--remote-log-dir",
        default="/vePFS-Mindverse/share/tmp/glm5_finegrained_fp8_quant_logs",
    )
    return p.parse_args()


def _make_env() -> dict[str, str]:
    return {
        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": (
            "/vePFS-Mindverse/share/code/mint-runtime-py31213-rl-v4/site-packages:"
            "/vePFS-Mindverse/share/code/tinker-server-aliyun:"
            "/vePFS-Mindverse/share/huggingface/modules:"
            "/vePFS-Mindverse/share/huggingface/modules/transformers_modules"
        ),
    }


def main() -> None:
    args = parse_args()
    ray.init(address=args.ray_address, namespace=args.ray_namespace, ignore_reinit_error=True)
    actor = NodeProcActor.options(
        num_gpus=8,
        num_cpus=8,
        resources={f"node:{args.node_ip}": 0.001},
    ).remote()
    actual_ip = ray.get(actor.node_ip.remote())
    if actual_ip != args.node_ip:
        raise RuntimeError(f"node pin mismatch: wanted {args.node_ip}, got {actual_ip}")

    env = _make_env()
    probe = ray.get(actor.probe_imports.remote(args.python, env))
    if probe["returncode"] != 0:
        raise RuntimeError(f"import probe failed: {probe}")
    payload = json.loads(str(probe["stdout"]).strip())
    missing = [name for name, ok in payload["mods"].items() if not ok]
    if missing:
        raise RuntimeError(f"missing imports on {args.node_ip}: {missing} payload={payload}")
    print(json.dumps({"phase": "probe_ok", "node_ip": args.node_ip, **payload}, ensure_ascii=True), flush=True)

    timestamp = int(time.time())
    log_path = str(Path(args.remote_log_dir) / f"glm5_finegrained_fp8_quant_node_{timestamp}.log")
    cmd = (
        f"{shlex.quote(args.python)} {shlex.quote(args.quant_script)} "
        f"--base-model-path {shlex.quote(args.base_model_path)} "
        f"--export-dir {shlex.quote(args.export_dir)} "
        "--trust-remote-code"
    )
    start = ray.get(actor.start.remote(cmd, env, args.workdir, log_path))
    print(json.dumps({"phase": "started", "node_ip": args.node_ip, **start}, ensure_ascii=True), flush=True)

    offset = 0
    while True:
        status = ray.get(actor.poll.remote())
        chunk = ray.get(actor.read_from.remote(offset))
        offset = int(chunk["end"])
        text = str(chunk["text"])
        if text:
            print(json.dumps({"phase": "log", "node_ip": args.node_ip, "text": text[-8000:]}, ensure_ascii=True), flush=True)
        print(json.dumps({"phase": "status", "node_ip": args.node_ip, **status}, ensure_ascii=True), flush=True)
        if not status.get("running", False):
            break
        time.sleep(15)

    status = ray.get(actor.poll.remote())
    if status.get("returncode") != 0:
        raise RuntimeError(f"finegrained fp8 quantization failed: {status}")
    print(json.dumps({"phase": "done", "node_ip": args.node_ip, **status}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
