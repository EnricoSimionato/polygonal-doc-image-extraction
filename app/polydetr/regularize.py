"""Regolarizzazione dei poligoni predetti: proiezione sulla famiglia di forme dei bersagli.

I bersagli del dataset v4 sono tutti convessi e regolari per costruzione (make_pages_v2.regular_target):
rettangolo, rettangolo ruotato, ellisse (24-gono), ottagono o poligono convesso con al più 8 lati. Il modello
predice invece 32 punti con rumore indipendente, e il rumore produce lati ondulati e vertici superflui. Qui si
sostituisce la predizione con la forma più semplice della famiglia che ne segue il bordo entro una tolleranza:

    rettangolo (minAreaRect) -> quadrilatero (prospettiva) -> ellisse -> 5..8 lati -> inviluppo convesso

e si tiene la predizione originale solo se nemmeno l'inviluppo convesso rientra nella tolleranza. Il criterio è
sulla distanza fra i bordi e non sull'IoU: un ottagono con smussi piccoli e il pentagono che li ignora hanno IoU
0.97, ma il bordo si sposta di 14 px negli angoli (verificato sui casi di prova). Distanza: P90 delle distanze
punto-bordo in entrambe le direzioni, così un candidato non può né tagliare né aggiungere un pezzo. Ogni candidato
si confronta con la predizione, non con la verità: la regolarizzazione funziona anche sulle foto reali.

Uso: from polydetr.regularize import regularize; poly, shape = regularize(pts, tol_px=2.0)
"""
from __future__ import annotations

import cv2
import numpy as np


def _mask(P: np.ndarray, origin: np.ndarray, s: float, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round((P - origin) * s * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    return m.astype(bool)


def iou(P: np.ndarray, Q: np.ndarray, res: int = 256) -> float:
    """IoU di due poligoni rasterizzati insieme su una griglia locale di lato massimo res (sub-pixel con shift=2)."""
    pts = np.vstack([P, Q])
    origin = pts.min(0) - 1
    ext = pts.max(0) + 1 - origin
    s = res / max(float(ext.max()), 1e-6)
    h, w = int(ext[1] * s) + 2, int(ext[0] * s) + 2
    a, b = _mask(P, origin, s, h, w), _mask(Q, origin, s, h, w)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _densify(P: np.ndarray, n: int = 96) -> np.ndarray:
    """n punti equispaziati lungo il contorno chiuso di P."""
    E = np.roll(P, -1, 0) - P
    L = np.linalg.norm(E, axis=1)
    c = np.r_[0, np.cumsum(L)]
    t = np.linspace(0, c[-1], n, endpoint=False)
    i = np.clip(np.searchsorted(c, t, side="right") - 1, 0, len(P) - 1)
    return P[i] + E[i] * ((t - c[i]) / np.maximum(L[i], 1e-9))[:, None]


def _dist_to_boundary(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Distanza di ogni punto di X dal contorno chiuso di P."""
    A, B = P, np.roll(P, -1, 0)
    AB = B - A
    t = np.clip(((X[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-9)[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(X[:, None] - proj, axis=-1).min(1)


def boundary_dev(Q: np.ndarray, P: np.ndarray) -> float:
    """Scarto fra i bordi di Q e P (pixel), simmetrico: il massimo fra il P90 e un terzo della distanza massima.
    Il P90 da solo lascia passare un angolo smussato mancante (pochi punti, molto lontani: un ottagono diventava un
    ettagono); il massimo da solo è dominato dal rumore."""
    d = np.r_[_dist_to_boundary(_densify(P, 192), Q), _dist_to_boundary(_densify(Q, 192), P)]
    return float(max(np.percentile(d, 90), d.max() / 3))


def outer_kgon(hull: np.ndarray, k: int) -> np.ndarray:
    """Poligono convesso con al più k vertici che contiene l'inviluppo: si toglie ogni volta il vertice la cui
    rimozione (prolungando i lati vicini) aggiunge meno area. Stessa regola di make_pages_v2.outer_kgon."""
    P = [np.asarray(p, float) for p in hull]
    while len(P) > k:
        best, best_i, best_pt = None, None, None
        m = len(P)
        for i in range(m):
            a, b, c, d = P[i - 2], P[i - 1], P[i], P[(i + 1) % m]
            A = np.array([b - a, d - c]).T
            if abs(np.linalg.det(A)) < 1e-9:
                continue
            t, u = np.linalg.solve(A, c - b)
            if t < 0 or u < 0:
                continue
            x = b + t * (b - a)
            area = 0.5 * abs((x - b)[0] * (c - b)[1] - (x - b)[1] * (c - b)[0])
            if best is None or area < best:
                best, best_i, best_pt = area, i, x
        if best is None:
            break
        P[best_i - 1] = best_pt
        del P[best_i]
    return np.array(P)


def inner_kgon(hull: np.ndarray, k: int) -> np.ndarray | None:
    """Approssimazione di Douglas-Peucker con esattamente k vertici (ricerca binaria su epsilon), o None."""
    h32 = hull.astype(np.float32).reshape(-1, 1, 2)
    per = cv2.arcLength(h32, True)
    lo, hi = 0.0, 0.2 * per
    for _ in range(30):
        eps = (lo + hi) / 2
        n = len(cv2.approxPolyDP(h32, eps, True))
        if n == k:
            return cv2.approxPolyDP(h32, eps, True)[:, 0].astype(float)
        lo, hi = (eps, hi) if n > k else (lo, eps)
    return None


def ellipse24(hull: np.ndarray) -> np.ndarray | None:
    if len(hull) < 5:
        return None
    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(hull.astype(np.float32))
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    x, y = ax1 / 2 * np.cos(t), ax2 / 2 * np.sin(t)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def refine(Q: np.ndarray, P: np.ndarray, keep_rect: bool = False) -> np.ndarray:
    """Toglie la distorsione verso l'esterno dei candidati costruiti sull'inviluppo (che contiene i punti più
    esterni del rumore): ogni lato si sposta sulla retta dei punti della predizione più vicini a quel lato, con un
    fit ai minimi quadrati totali, e i nuovi vertici sono le intersezioni fra rette consecutive. Con keep_rect il
    lato si trasla soltanto (mediana degli scarti), così un rettangolo resta un rettangolo."""
    X = _densify(P, 192)
    A, B = Q, np.roll(Q, -1, 0)
    AB = B - A
    t = np.clip(((X[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-9)[None], 0, 1)
    dist = np.linalg.norm(X[:, None] - (A[None] + t[..., None] * AB[None]), axis=-1)
    own = dist.argmin(1)
    lines = []
    for e in range(len(Q)):
        d = AB[e] / max(np.linalg.norm(AB[e]), 1e-9)
        n = np.array([-d[1], d[0]])
        pts = X[(own == e) & (t[np.arange(len(X)), e] > 0.1) & (t[np.arange(len(X)), e] < 0.9)]
        if len(pts) < 3:
            lines.append((A[e], d))
            continue
        if keep_rect:
            lines.append((A[e] + n * float(np.median((pts - A[e]) @ n)), d))
        else:
            c = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - c)
            d2 = vt[0] if vt[0] @ d >= 0 else -vt[0]
            lines.append((c, d2))
    out = []
    for e in range(len(Q)):
        (p1, d1), (p2, d2) = lines[e - 1], lines[e]
        M = np.array([d1, -d2]).T
        if abs(np.linalg.det(M)) < 1e-6:
            return Q
        s_, _ = np.linalg.solve(M, p2 - p1)
        out.append(p1 + s_ * d1)
    out = np.array(out)
    # una retta sbagliata può far esplodere un vertice: in quel caso si tiene il candidato
    return out if np.abs(out - Q).max() < 0.25 * np.ptp(Q, 0).max() else Q


def regularize(P: np.ndarray, tol_px: float = 2.0, tol_rel: float = 0.015) -> tuple[np.ndarray, str]:
    """Restituisce (poligono, nome della forma). P: (N, 2) in pixel, in ordine lungo il contorno.
    Tolleranza: max(tol_px, tol_rel * sqrt(area)), così scala con l'oggetto ma non scende sotto il rumore."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P, "orig"
    hull = cv2.convexHull(P.astype(np.float32))[:, 0].astype(float)
    if len(hull) < 3:
        return P, "orig"
    tol = max(tol_px, tol_rel * np.sqrt(max(cv2.contourArea(hull.astype(np.float32)), 0.0)))

    steps = [("rect", lambda: [cv2.boxPoints(cv2.minAreaRect(hull.astype(np.float32))).astype(float)]),
             ("quad", lambda: [outer_kgon(hull, 4), inner_kgon(hull, 4)]),
             ("ellipse", lambda: [ellipse24(hull)])]
    steps += [(f"{k}-gon", (lambda k=k: [outer_kgon(hull, k), inner_kgon(hull, k)])) for k in range(5, 9)]
    steps += [("hull", lambda: [hull])]
    for name, make in steps:
        cands = [c for c in make() if c is not None and len(c) >= 3]
        if name not in ("ellipse", "hull"):
            cands = [refine(c, P, keep_rect=(name == "rect")) for c in cands]
        if not cands:
            continue
        devs = [boundary_dev(c, P) for c in cands]
        j = int(np.argmin(devs))
        if devs[j] <= tol:
            return cands[j], name
    return P, "orig"
