#!/usr/bin/env python3
"""Serve the birdsong demo with the Chapter 1 pipeline behind it, on this computer.

    python examples/pipeline_server.py            # then open http://127.0.0.1:8000
    python examples/pipeline_server.py --port 8010

The page posts the WAV it is playing (or an uploaded .wav) to /api/run. The
server runs examples/pipeline_runner.run_pipeline on those bytes in a
background thread and the page polls /api/jobs/<id> for GA progress and the
result. Only one pipeline run happens at a time; the GA is CPU-bound.
Requires the C++ extensions (bash scripts/build_cpp_extensions.sh).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pipeline_runner

PAGE = Path(__file__).resolve().parent / "birdsong_demo.html"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

jobs: dict[str, dict] = {}
run_lock = threading.Lock()


def run_job(job_id: str, wav_bytes: bytes, seed: int) -> None:
    job = jobs[job_id]
    try:
        with tempfile.TemporaryDirectory() as scratch:
            wav_path = Path(scratch) / "input.wav"
            wav_path.write_bytes(wav_bytes)
            job["result"] = pipeline_runner.run_pipeline(
                wav_path, seed, on_progress=lambda p: job.update(progress=p))
            job["result"]["wav"] = "uploaded audio"
        job["status"] = "done"
    except Exception as error:  # report any failure to the page
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = f"{type(error).__name__}: {error}"
    finally:
        run_lock.release()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/health":
            self.send_json({"pipeline": True, "config": pipeline_runner.CONFIG_SCRIPT,
                            "references": pipeline_runner.REFERENCES_JSON.exists()})
        elif url.path.startswith("/api/jobs/"):
            job = jobs.get(url.path.rsplit("/", 1)[-1])
            if job is None:
                self.send_json({"error": "unknown job"}, 404)
            else:
                self.send_json(job)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/run":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 44 < length <= MAX_UPLOAD_BYTES:
            self.send_json({"error": "Send a WAV file of at most 20 MB."}, 400)
            return
        wav_bytes = self.rfile.read(length)
        if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            self.send_json({"error": "The upload is not a WAV file."}, 400)
            return
        try:
            seed = int(parse_qs(url.query).get("seed", ["47"])[0])
        except ValueError:
            self.send_json({"error": "The seed must be an integer."}, 400)
            return
        if not run_lock.acquire(blocking=False):
            self.send_json({"error": "A pipeline run is already in progress."}, 409)
            return
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "running", "progress": None, "seed": seed}
        threading.Thread(target=run_job, args=(job_id, wav_bytes, seed), daemon=True).start()
        self.send_json({"job": job_id})

    def log_message(self, fmt, *args):
        if "/api/jobs/" not in self.path:  # polling would flood the console
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    pipeline_runner.pipeline_module()  # fail early if the C++ extensions are missing
    if not pipeline_runner.REFERENCES_JSON.exists():
        print("No reference classes yet; run python examples/build_references.py for class labels.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Birdsong pipeline demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
