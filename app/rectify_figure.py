"""Da poligono a immagine rettangolare: due algoritmi deterministici di completamento (idee di Enrico, 2026-10-17).

Un'immagine estratta con un contorno libero (sagoma, figura con richiami, disegno senza cornice) non è un rettangolo,
ma chi la usa dopo si aspetta un rettangolo. La cornice è il **rettangolo di contorno del poligono allineato agli assi
dell'immagine in ingresso**: l'immagine estratta non si ruota e non si raddrizza (Enrico, 2026-10-18). La parte della
cornice fuori dal poligono si riempie:

- `rectify_figure`: con il colore della PAGINA, letto nell'anello di pixel subito fuori dal poligono; se almeno
  `uniform_frac` dell'anello sta entro `tol` livelli per canale dal colore mediano si usa quel colore, altrimenti il
  bianco;
- `complete_figure`: con il fondo della FIGURA, letto sul bordo interno del poligono (Delta E in Lab) e propagato
  fuori con un inpainting di Telea, così un gradiente leggero prosegue; bianco se il bordo non è omogeneo.

Le cornici ruotate (rettangolo di area minima) e il raddrizzamento prospettico dei quadrilateri restano disponibili
con `frame="minima"` e `frame="prospettiva"`, ma non sono il comportamento predefinito.

Uso come modulo: rectify_figure(img_rgb, poly) e complete_figure(img_rgb, poly) -> (immagine RGB, info).
Uso da riga di comando: python rectify_figure.py --image foto.jpg --polys poligoni.json --out cartella
(poligoni.json: lista di liste di [x, y] in pixel della foto).
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def _order_quad(Q: np.ndarray) -> np.ndarray:
    """Vertici in ordine alto-sinistra, alto-destra, basso-destra, basso-sinistra."""
    c = Q.mean(0)
    Q = Q[np.argsort(np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0]))]
    return np.roll(Q, -int(np.argmin(Q.sum(1))), axis=0)


def _frame(img: np.ndarray, P: np.ndarray, frame: str = "assi", quad_tol: float = 0.02):
    """-> (immagine nella cornice, maschera del poligono nella cornice, modo).

    assi: rettangolo di contorno allineato all'immagine in ingresso, nessuna rotazione (predefinito);
    minima: rettangolo di area minima, raddrizzato con la rotazione più piccola;
    prospettiva: quadrilatero quasi esatto raddrizzato con un'omografia, altrimenti come minima."""
    h0, w0 = img.shape[:2]
    if frame == "assi":
        x0, y0 = np.floor(P.min(0)).astype(int).clip(0, [w0 - 1, h0 - 1])
        x1, y1 = np.ceil(P.max(0)).astype(int).clip(1, [w0, h0])
        out = img[y0:y1, x0:x1].copy()
        mask = np.zeros(out.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(P - [x0, y0]).astype(np.int32)], 1)
        return out, mask, "assi"
    if frame == "prospettiva":
        area = float(cv2.contourArea(P))
        hull = cv2.convexHull(P)
        q = cv2.approxPolyDP(hull, quad_tol * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float32)
        if (len(q) == 4 and area / max(float(cv2.contourArea(hull)), 1e-6) > 0.97
                and area / max(float(cv2.contourArea(q)), 1e-6) > 0.95):
            Q = _order_quad(q)
            w = int(round((np.linalg.norm(Q[1] - Q[0]) + np.linalg.norm(Q[2] - Q[3])) / 2))
            h = int(round((np.linalg.norm(Q[3] - Q[0]) + np.linalg.norm(Q[2] - Q[1])) / 2))
            if w >= 2 and h >= 2:
                M = cv2.getPerspectiveTransform(Q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
                out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return out, np.ones((h, w), np.uint8), "prospettiva"
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(P)
    if ang > 45:
        ang, rw, rh = ang - 90, rh, rw
    elif ang < -45:
        ang, rw, rh = ang + 90, rh, rw
    w, h = max(int(np.ceil(rw)), 2), max(int(np.ceil(rh)), 2)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[:, 2] += (w / 2 - cx, h / 2 - cy)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(cv2.transform(P[None], M)[0]).astype(np.int32)], 1)
    return out, mask, "minima"


def background_color(img: np.ndarray, poly: np.ndarray, ring: int, tol: float = 24.0, uniform_frac: float = 0.7):
    """Colore mediano dell'anello largo `ring` px fuori dal poligono e se è uniforme. -> (colore RGB, uniforme, quota)"""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    px = img[(cv2.dilate(m, k) > 0) & (m == 0)]
    if len(px) < 20:
        return (255, 255, 255), False, 0.0
    med = np.median(px, axis=0)
    frac = float((np.abs(px.astype(np.float32) - med).max(1) <= tol).mean())
    uniform = frac >= uniform_frac
    return (tuple(int(v) for v in med) if uniform else (255, 255, 255)), uniform, frac


def rectify_figure(img: np.ndarray, poly, tol: float = 24.0, uniform_frac: float = 0.7, max_side: int | None = None,
                   frame: str = "assi"):
    """Primo algoritmo: cornice riempita con il colore uniforme della pagina attorno al poligono, o il bianco."""
    P = np.asarray(poly, np.float32).reshape(-1, 2)
    out, mask, mode = _frame(img, P, frame)
    info = {"mode": mode}
    if mask.all():
        info.update(fill=None, uniform=None)
        return _limit(out, max_side), info
    ring = max(4, int(round(0.02 * min(out.shape[:2]))))
    color, uniform, frac = background_color(img, P, ring, tol, uniform_frac)
    out[mask == 0] = color
    info.update(fill=list(color), uniform=uniform, ring_uniform_frac=round(frac, 3))
    return _limit(out, max_side), info


def _lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0                      # L di OpenCV in [0, 255] -> [0, 100]
    lab[..., 1:] -= 128.0
    return lab


def complete_figure(img: np.ndarray, poly, de_tol: float = 12.0, uniform_frac: float = 0.6, work_side: int = 320,
                    max_side: int | None = None, frame: str = "assi"):
    """Secondo algoritmo: la cornice si colora con il fondo della FIGURA, letto sul suo bordo interno.

    1. bordo interno: pixel del poligono entro b = max(3 px, 1.5% del lato corto) dal contorno;
    2. omogeneità in Lab: se almeno `uniform_frac` del bordo sta entro `de_tol` (Delta E 76) dal colore mediano, il
       fondo è omogeneo e si accettano anche i colori vicini; altrimenti si riempie di bianco;
    3. riempimento: i pixel del bordo lontani dal mediano (contenuto che tocca il margine) diventano il mediano, poi i
       colori si propagano fuori dal poligono con un inpainting di Telea calcolato a `work_side` px di lato e riportato
       alla risoluzione piena; i colori propagati oltre 2 de_tol dal mediano tornano al mediano."""
    P = np.asarray(poly, np.float32).reshape(-1, 2)
    out, mask, mode = _frame(img, P, frame)
    info = {"mode": mode}
    if mask.all():
        info.update(fill=None, uniform=None)
        return _limit(out, max_side), info
    h, w = mask.shape
    b = max(3, int(round(0.015 * min(h, w))))
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1)))
    band = (mask > 0) & (inner == 0)
    lab = _lab(out)
    if band.sum() < 20:
        out[mask == 0] = 255
        info.update(fill=[255, 255, 255], uniform=False, border_uniform_frac=0.0)
        return _limit(out, max_side), info
    med = np.median(lab[band], axis=0)
    de = np.linalg.norm(lab - med, axis=2)
    frac = float((de[band] <= de_tol).mean())
    if frac < uniform_frac:
        out[mask == 0] = 255
        info.update(fill=[255, 255, 255], uniform=False, border_uniform_frac=round(frac, 3))
        return _limit(out, max_side), info
    med_rgb = np.median(out[band & (de <= de_tol)], axis=0).astype(np.uint8)
    src = out.copy()
    src[(mask > 0) & (de > de_tol * 1.5)] = med_rgb
    s = min(1.0, work_side / max(h, w))
    sw, sh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
    hole = cv2.dilate(cv2.resize((mask == 0).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST),
                      np.ones((3, 3), np.uint8))
    big = cv2.resize(cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC)
    big[np.linalg.norm(_lab(big) - med, axis=2) > 2 * de_tol] = med_rgb
    out[mask == 0] = big[mask == 0]
    info.update(fill=[int(v) for v in med_rgb], uniform=True, border_uniform_frac=round(frac, 3))
    return _limit(out, max_side), info


def _limit(img: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side or max(img.shape[:2]) <= max_side:
        return img
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--polys", required=True, help="JSON: lista di poligoni [[x, y], ...] in pixel della foto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metodo", choices=["pagina", "bordo"], default="bordo",
                    help="pagina: rectify_figure (colore della pagina); bordo: complete_figure (fondo della figura)")
    ap.add_argument("--cornice", choices=["assi", "minima", "prospettiva"], default="assi")
    a = ap.parse_args(argv)
    img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    os.makedirs(a.out, exist_ok=True)
    fn = complete_figure if a.metodo == "bordo" else rectify_figure
    for i, P in enumerate(json.load(open(a.polys))):
        out, info = fn(img, P, frame=a.cornice)
        cv2.imwrite(os.path.join(a.out, f"figura_{i:02d}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(i, info)


if __name__ == "__main__":
    main()
