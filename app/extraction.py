"""Image extraction from document pages and photos with PolyDETR, on CPU.

For every page: polygon prediction (low threshold, 0.2; the UI filters by score), a polygonal cut-out with a
transparent background, and on demand three rectangular versions filled with the figure's own border colour
(rectify_figure.complete_figure): axis-aligned, straightened (minimum-area rotated frame) or perspective-corrected.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms

from polydetr.losses import prune
from polydetr.model_v2 import build_model
from polydetr.regularize import regularize as regularize_poly
from rectify_figure import complete_figure

try:  # HEIC photos from iPhones
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = 120_000_000   # refuses absurdly large images (decompression bombs)
MAX_SIDE = 2000                        # pages and photos are processed with the long side at most this many pixels
PDF_DPI = 150
MAX_PAGES = 300                        # per upload

WEIGHTS_URL = ("https://github.com/EnricoSimionato/polygonal-doc-image-extraction/releases/download/"
               "v0/polydetr-v0.safetensors")

# extraction modes: (rectangle frame for complete_figure) or None for the polygonal cut-out
MODES = {"polygon": None, "filled": "assi", "straightened": "minima", "perspective": "prospettiva"}


def ensure_weights(weights_dir: Path) -> Path:
    """Path of the weights, downloaded from the v0 release on first use and checked against SHA256SUMS."""
    name = "polydetr-v0.safetensors"
    expected = (weights_dir / "SHA256SUMS").read_text().split()[0]
    path = weights_dir / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        print(f"Downloading the model weights (93 MB) from {WEIGHTS_URL} ...", flush=True)
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != expected:
        path.unlink()
        raise RuntimeError(f"checksum mismatch for {name}: expected {expected}, got {h}. The file was removed; "
                           "restart to download it again.")
    return path


def load_model(weights_dir: Path):
    from safetensors.torch import load_file      # no pickle: loading the weights cannot execute code
    cfg = json.loads((weights_dir / "polydetr-v0.json").read_text())
    m = build_model(cfg)
    m.load_state_dict(load_file(str(ensure_weights(weights_dir))))
    return m.eval(), cfg


def letterbox(img: np.ndarray, H: int, W: int, fill: int = 114):
    h0, w0 = img.shape[:2]
    s = min(H / h0, W / w0)
    nh, nw = max(int(round(h0 * s)), 1), max(int(round(w0 * s)), 1)
    canvas = np.full((H, W, 3), fill, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, s


@torch.inference_mode()
def predict(model, cfg, img: np.ndarray, thr: float = 0.2, nms_iou: float = 0.7, regularize: float = 1.0):
    """-> list of (score, polygon (K, 2) in image pixels), highest score first."""
    H, W = cfg["height"], cfg["width"]
    canvas, s = letterbox(img, H, W)
    x = torch.from_numpy(canvas).permute(2, 0, 1).float().div(255)[None]
    o = model(x)
    sc = o["logits"][0].float().sigmoid()
    pts = o["points"][0].float()
    keep = nms(torch.cat([pts.min(-2).values, pts.max(-2).values], dim=-1), sc, nms_iou)
    scale = torch.tensor([W, H], dtype=torch.float32)
    out = []
    for q in keep:
        if sc[q] < thr:
            continue
        P = prune(pts[q] * scale).numpy()
        if regularize > 0:
            P = regularize_poly(P, tol_px=regularize)[0]
        out.append((float(sc[q]), np.asarray(P, float) / s))
    return out


def crop_polygon(img: np.ndarray, poly: np.ndarray, ss: int = 4) -> np.ndarray:
    """RGBA: bounding box of the polygon, alpha = anti-aliased polygon coverage."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(poly.min(0)).astype(int).clip(0, [w - 1, h - 1])
    x1, y1 = np.ceil(poly.max(0)).astype(int).clip(1, [w, h])
    m = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    cv2.fillPoly(m, [np.round((poly - [x0, y0]) * ss * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return np.dstack([img[y0:y1, x0:x1], cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)])


def _limit(img: np.ndarray) -> np.ndarray:
    if max(img.shape[:2]) <= MAX_SIDE:
        return img
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)


def split_pages(name: str, data: bytes):
    """Uploaded file -> (label, RGB image) for every page. PDFs are rendered with PDFium at PDF_DPI; multi-page
    TIFFs give one page per frame; other images give one page, rotated according to EXIF."""
    if data[:5] == b"%PDF-":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
        n = len(pdf)
        for i in range(min(n, MAX_PAGES)):
            img = np.asarray(pdf[i].render(scale=PDF_DPI / 72).to_pil().convert("RGB")).copy()
            yield f"{name} · page {i + 1} of {n}", _limit(img)
        return
    im = Image.open(io.BytesIO(data))
    n = getattr(im, "n_frames", 1)
    for i in range(min(n, MAX_PAGES)):
        if n > 1:
            im.seek(i)
        fr = ImageOps.exif_transpose(im.convert("RGB")) if n == 1 else im.convert("RGB")
        yield (f"{name} · page {i + 1} of {n}" if n > 1 else name), _limit(np.asarray(fr).copy())


def save_png(path: Path, img: np.ndarray):
    code = cv2.COLOR_RGBA2BGRA if img.shape[2] == 4 else cv2.COLOR_RGB2BGR
    cv2.imwrite(str(path), cv2.cvtColor(img, code))


def render_mode(img: np.ndarray, poly, mode: str) -> np.ndarray:
    """One extracted image in the requested mode (see MODES)."""
    P = np.asarray(poly, np.float32)
    if MODES[mode] is None:
        return crop_polygon(img, P)
    out, _ = complete_figure(img, P, frame=MODES[mode])
    return out


def process_page(model, cfg, img: np.ndarray, out_dir: Path, stem: str) -> dict:
    """Predicts and saves the page and the polygonal cut-outs; the other modes are rendered on request."""
    cv2.imwrite(str(out_dir / f"{stem}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    figs = []
    h, w = img.shape[:2]
    for k, (score, P) in enumerate(predict(model, cfg, img)):
        P[:, 0] = P[:, 0].clip(0, w - 1)
        P[:, 1] = P[:, 1].clip(0, h - 1)
        if cv2.contourArea(P.astype(np.float32)) < 64:
            continue
        figs.append({"score": round(score, 3), "poly": np.round(P, 1).tolist()})
    return {"image": f"{stem}.jpg", "w": int(w), "h": int(h), "figures": figs}
