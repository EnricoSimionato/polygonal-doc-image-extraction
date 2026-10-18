"""Local web app for polygonal document image extraction. Start it with run.command / run.sh / run.bat.

Security model: the server listens on 127.0.0.1 only, accepts requests only with a Host header for 127.0.0.1 or
localhost (against DNS rebinding), and every API call must carry a random token generated at start-up and embedded
in the page it serves (other web sites open in the browser cannot read it, so they cannot drive the API). Uploaded
files and results live in a temporary folder that is deleted when the app stops; nothing leaves the computer.
"""
from __future__ import annotations

import atexit
import io
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extraction  # noqa: E402

WEIGHTS = HERE.parent / "weights"
WORK = Path(tempfile.mkdtemp(prefix="polygonal-doc-image-extraction-"))
atexit.register(shutil.rmtree, WORK, True)
TOKEN = secrets.token_urlsafe(24)
JOB_ID = re.compile(r"^[0-9a-f]{16}$")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
JOBS: dict[str, dict] = {}
MODEL = {"m": None, "cfg": None, "error": None}
CPU = threading.Lock()
PORT = {"n": None}


def _load_model():
    try:
        MODEL["m"], MODEL["cfg"] = extraction.load_model(WEIGHTS)
    except Exception as e:                     # shown in the UI
        MODEL["error"] = f"{type(e).__name__}: {e}"


@app.before_request
def _guard():
    host = request.host.split(":")[0]
    if host not in ("127.0.0.1", "localhost"):
        abort(403)
    if request.path.startswith("/api/") and secrets.compare_digest(
            request.headers.get("X-Token", request.args.get("t", "")), TOKEN) is False:
        abort(403)


def _job(job_id: str) -> dict:
    if not JOB_ID.match(job_id) or job_id not in JOBS:
        abort(404)
    return JOBS[job_id]


def _run(job_id: str, files: list[tuple[str, bytes]]):
    job, out = JOBS[job_id], WORK / job_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        pages = []
        for name, data in files:
            try:
                pages += list(extraction.split_pages(name, data))
            except Exception as e:
                job["warnings"].append(f"{name}: could not be read ({type(e).__name__})")
        job["total"] = len(pages)
        while MODEL["m"] is None and MODEL["error"] is None:
            time.sleep(0.3)
        if MODEL["error"]:
            raise RuntimeError(MODEL["error"])
        for i, (label, img) in enumerate(pages):
            t0 = time.time()
            with CPU:
                pg = extraction.process_page(MODEL["m"], MODEL["cfg"], img, out, f"p{i:04d}")
            pg.update(label=label, seconds=round(time.time() - t0, 2))
            job["pages"].append(pg)
            job["done"] = i + 1
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


def _figure_png(job_id: str, i: int, k: int, mode: str) -> Path:
    """The extracted image in the requested mode, rendered on first request and cached."""
    job = _job(job_id)
    if mode not in extraction.MODES or not (0 <= i < len(job["pages"])) or not (0 <= k < len(job["pages"][i]["figures"])):
        abort(404)
    pg = job["pages"][i]
    path = WORK / job_id / f"p{i:04d}_{k:02d}_{mode}.png"
    if not path.exists():
        import cv2
        img = cv2.cvtColor(cv2.imread(str(WORK / job_id / pg["image"])), cv2.COLOR_BGR2RGB)
        extraction.save_png(path, extraction.render_mode(img, pg["figures"][k]["poly"], mode))
    return path


@app.get("/")
def index():
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
    return html, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store",
                       "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; "
                                                  "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"}


@app.get("/api/status")
def status():
    return jsonify({"ready": MODEL["m"] is not None, "error": MODEL["error"]})


@app.post("/api/jobs")
def new_job():
    files = [(os.path.basename(f.filename or "file"), f.read()) for f in request.files.getlist("files")]
    files = [(n, d) for n, d in files if d]
    if not files:
        return jsonify({"error": "no files"}), 400
    job_id = secrets.token_hex(8)
    JOBS[job_id] = {"id": job_id, "status": "running", "done": 0, "total": None, "pages": [], "warnings": [],
                    "files": [n for n, _ in files], "error": None}
    threading.Thread(target=_run, args=(job_id, files), daemon=True).start()
    return jsonify({"id": job_id})


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    return jsonify(_job(job_id))


@app.get("/api/jobs/<job_id>/page/<int:i>.jpg")
def page_image(job_id, i):
    job = _job(job_id)
    if not 0 <= i < len(job["pages"]):
        abort(404)
    return send_file(WORK / job_id / job["pages"][i]["image"], mimetype="image/jpeg")


@app.get("/api/jobs/<job_id>/figure/<int:i>/<int:k>/<mode>.png")
def figure_image(job_id, i, k, mode):
    return send_file(_figure_png(job_id, i, k, mode), mimetype="image/png",
                     download_name=f"page-{i + 1:03d}-image-{k + 1:02d}-{mode}.png")


@app.get("/api/jobs/<job_id>/zip")
def zip_job(job_id):
    """ZIP of the images shown in the UI: mode, min_score and optionally a single page."""
    job = _job(job_id)
    mode = request.args.get("mode", "filled")
    if mode not in extraction.MODES:
        abort(400)
    min_score = float(request.args.get("min_score", 0.4))
    only = request.args.get("page")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i, pg in enumerate(job["pages"]):
            if only is not None and str(i) != only:
                continue
            kept = [k for k, f in enumerate(pg["figures"]) if f["score"] >= min_score]
            for n, k in enumerate(kept):
                z.write(_figure_png(job_id, i, k, mode), f"page-{i + 1:03d}/image-{n + 1:02d}-{mode}.png")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"extracted-images-{mode}.zip")


def _free_port(start=8765):
    for p in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise SystemExit("no free port between 8765 and 8814")


if __name__ == "__main__":
    threading.Thread(target=_load_model, daemon=True).start()
    PORT["n"] = _free_port()
    url = f"http://127.0.0.1:{PORT['n']}/"
    print(f"\nPolygonal document image extraction is running at {url}\nClose this window (or press Ctrl+C) to stop it; "
          "uploaded files and results are deleted when it stops.\n", flush=True)
    if os.environ.get("PDIE_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT["n"], threaded=True, debug=False)
