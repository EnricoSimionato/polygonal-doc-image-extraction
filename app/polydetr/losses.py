"""Loss per poligoni a vertici variabili: distanza fra contorni e penalità sugli angoli di svolta.

Notazione: un poligono è un tensore V di forma (N, 2), punti (x, y) in pixel, ordinati lungo il
contorno e chiuso implicitamente (l'ultimo punto si collega al primo).

- `contour_loss`: distanza di Chamfer fra il contorno predetto e quello vero, nelle due
  direzioni (vedi DESIGN-NOTES, passo 4). Non confronta vertici con vertici.
- `turning_angles`: angolo di cui gira la direzione del contorno in ogni punto, in radianti.
- `corner_penalty`: somma di (theta^2 + delta^2)^(p/2) - delta^p, versione liscia di |theta|^p.
  Con p < 1 è subadditiva, quindi favorisce pochi angoli grandi (pochi vertici).
- `prune`: toglie i punti con angolo di svolta quasi nullo e restituisce i vertici veri.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_polygon(V: torch.Tensor, per_edge: int = 16):
    """Campiona per_edge punti su ogni lato. Restituisce i punti (N*per_edge, 2) e il peso di
    ciascuno (lunghezza del lato / per_edge), così la media è una media lungo l'arco."""
    W = torch.roll(V, -1, dims=0)
    t = (torch.arange(per_edge, device=V.device, dtype=V.dtype) + 0.5) / per_edge
    pts = V[:, None, :] * (1 - t)[None, :, None] + W[:, None, :] * t[None, :, None]
    w = (W - V).norm(dim=1, keepdim=True).expand(-1, per_edge) / per_edge
    return pts.reshape(-1, 2), w.reshape(-1)


def sample_map(D: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Legge la mappa D (H, W) nei punti pts (M, 2) in pixel, con interpolazione bilineare."""
    H, W = D.shape
    g = torch.stack([2 * (pts[:, 0] + 0.5) / W - 1, 2 * (pts[:, 1] + 0.5) / H - 1], dim=1)
    out = F.grid_sample(D[None, None], g[None, None], mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.reshape(-1)


def point_segment_distance(P: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Distanza di ogni punto P (G, 2) dal lato più vicino del poligono V (N, 2)."""
    A, B = V, torch.roll(V, -1, dims=0)
    AB = B - A                                            # (N, 2)
    AP = P[:, None, :] - A[None, :, :]                    # (G, N, 2)
    denom = (AB * AB).sum(-1).clamp_min(1e-9)             # (N,)
    t = ((AP * AB[None]).sum(-1) / denom[None]).clamp(0, 1)
    proj = A[None] + t[..., None] * AB[None]
    d = (P[:, None, :] - proj).norm(dim=-1)               # (G, N)
    return d.min(dim=1).values


def contour_loss(V: torch.Tensor, D_gt: torch.Tensor, gt_pts: torch.Tensor, per_edge: int = 16):
    """Media delle due direzioni: predetto -> vero (mappa delle distanze) e vero -> predetto
    (distanza punto-lato). Unità: pixel."""
    q, w = sample_polygon(V, per_edge)
    pred_to_gt = (sample_map(D_gt, q) * w).sum() / w.sum().clamp_min(1e-9)
    gt_to_pred = point_segment_distance(gt_pts, V).mean()
    return 0.5 * (pred_to_gt + gt_to_pred)


def turning_angles(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Angolo di svolta in ogni punto: l'angolo fra il lato entrante e quello uscente."""
    e_in = V - torch.roll(V, 1, dims=0)
    e_out = torch.roll(V, -1, dims=0) - V
    cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    dot = (e_in * e_out).sum(-1)
    return torch.atan2(cross, dot + eps)


def corner_penalty(V: torch.Tensor, p: float = 0.5, delta: float = 0.02) -> torch.Tensor:
    th = turning_angles(V)
    return ((th ** 2 + delta ** 2) ** (p / 2) - delta ** p).sum()


@torch.no_grad()
def prune(V: torch.Tensor, tau: float = 0.02, min_edge: float = 0.5) -> torch.Tensor:
    """Toglie iterativamente i punti con |angolo di svolta| < tau (radianti) e i punti che
    coincidono con il precedente; restituisce i vertici rimasti."""
    keep = V.clone()
    changed = True
    while changed and len(keep) > 3:
        changed = False
        e = (keep - torch.roll(keep, 1, dims=0)).norm(dim=1)
        th = turning_angles(keep).abs()
        bad = (th < tau) | (e < min_edge)
        if bad.any():
            i = int(torch.argmin(torch.where(bad, th, torch.full_like(th, 1e9))))
            keep = torch.cat([keep[:i], keep[i + 1:]])
            changed = True
    return keep


def contour_loss_poly(V: torch.Tensor, G: torch.Tensor, per_edge: int = 16) -> torch.Tensor:
    """Distanza simmetrica fra il poligono predetto V (N, 2) e il contorno vero G (M, 2),
    entrambi poligoni chiusi in pixel. Ogni direzione campiona punti lungo i lati di uno e
    misura la distanza dal lato più vicino dell'altro; media pesata sulla lunghezza d'arco.
    È la versione usata nel training: non servono mappe delle distanze."""
    q, w = sample_polygon(V, per_edge)
    pred_to_gt = (point_segment_distance(q, G) * w).sum() / w.sum().clamp_min(1e-9)
    g, wg = sample_polygon(G, 1)
    gt_to_pred = (point_segment_distance(g, V) * wg).sum() / wg.sum().clamp_min(1e-9)
    return 0.5 * (pred_to_gt + gt_to_pred)


def signed_area(V: torch.Tensor) -> torch.Tensor:
    """Area con segno (formula di Gauss): positiva o negativa secondo il verso di percorrenza.
    Un poligono che si ripiega su se stesso ha |area| più piccola di quella del contorno."""
    x, y = V[:, 0], V[:, 1]
    return 0.5 * (x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum()


def area_loss(V: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """|area(V)| contro |area(G)|, relativa. Il verso è libero: conta che il bordo sia
    percorso una volta sola, non in quale direzione."""
    a_g = signed_area(G).abs().clamp_min(1.0)
    return (signed_area(V).abs() - a_g).abs() / a_g


def resample_closed(G: torch.Tensor, n: int) -> torch.Tensor:
    """n punti equispaziati lungo il perimetro del poligono chiuso G."""
    W = torch.roll(G, -1, dims=0)
    seg = (W - G).norm(dim=1)
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    L = cum[-1].clamp_min(1e-9)
    s = torch.arange(n, device=G.device, dtype=G.dtype) * L / n
    idx = torch.searchsorted(cum[1:], s, right=True).clamp(max=len(G) - 1)
    t = ((s - cum[idx]) / seg[idx].clamp_min(1e-9)).unsqueeze(1)
    return G[idx] * (1 - t) + W[idx] * t


def ordered_loss(V: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Distanza media punto a punto fra V (N punti) e N punti equispaziati sul contorno vero,
    con il miglior spostamento ciclico e il miglior verso. Impone che V percorra il bordo una
    volta sola e in ordine. Si usa solo nelle prime epoche: a regime imporrebbe punti
    equispaziati e taglierebbe gli angoli."""
    n = V.shape[0]
    Gs = resample_closed(G, n)
    best = None
    for P in (Gs, torch.flip(Gs, dims=[0])):
        idx = (torch.arange(n, device=V.device)[None, :] + torch.arange(n, device=V.device)[:, None]) % n
        d = (V[None, :, :] - P[idx]).norm(dim=-1).mean(dim=1)  # (n spostamenti,)
        m = d.min()
        best = m if best is None else torch.minimum(best, m)
    return best


def is_simple(V) -> bool:
    """True se nessuna coppia di lati non adiacenti si interseca."""
    P = V.tolist() if hasattr(V, "tolist") else V
    n = len(P)

    def orient(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            a, b, c, d = P[i], P[(i + 1) % n], P[j], P[(j + 1) % n]
            if orient(c, d, a) * orient(c, d, b) < 0 and orient(a, b, c) * orient(a, b, d) < 0:
                return False
    return True


# ------------------------------------------------------------------ versioni vettoriali (training)
# Le stesse loss di sopra, calcolate su Z coppie (predetto, vero) in una volta sola. I contorni veri hanno
# lunghezze diverse: si allungano fino a M ripetendo l'ultimo vertice. Un vertice ripetuto crea un lato di
# lunghezza zero, che non sposta nessuna distanza (il suo punto sta già sul contorno), ha peso zero nelle
# medie d'arco e non cambia l'area di Gauss, quindi le loss non cambiano.

def pad_polygons(polys: list[torch.Tensor]) -> torch.Tensor:
    """Lista di (M_k, 2) -> (K, M, 2), ripetendo l'ultimo vertice."""
    M = max(len(p) for p in polys)
    return torch.stack([torch.cat([p, p[-1:].expand(M - len(p), 2)]) if len(p) < M else p for p in polys])


def point_segment_distance_b(P: torch.Tensor, V: torch.Tensor, max_elems: int = 8_000_000) -> torch.Tensor:
    """P (Z, G, 2), V (Z, N, 2) -> (Z, G): distanza di ogni punto dal lato più vicino del suo poligono.

    I tensori intermedi hanno Z·G·N elementi: con poligoni veri da centinaia di vertici superano i GB e, conservati
    per il backward in ogni strato del decoder, esauriscono la GPU (OOM del 2026-10-09, batch 12). Sopra
    max_elems il calcolo si fa a blocchi lungo Z, con activation checkpointing: stesso risultato e stesso gradiente,
    memoria limitata a un blocco."""
    Z, G, N = P.shape[0], P.shape[1], V.shape[1]
    if Z * G * N <= max_elems or Z <= 1:
        return _psd(P, V)
    from torch.utils.checkpoint import checkpoint

    step = max(1, max_elems // max(G * N, 1))
    grad = torch.is_grad_enabled() and (P.requires_grad or V.requires_grad)
    out = [checkpoint(_psd, P[i:i + step], V[i:i + step], use_reentrant=False) if grad else _psd(P[i:i + step], V[i:i + step])
           for i in range(0, Z, step)]
    return torch.cat(out)


def _psd(P: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    A, B = V, torch.roll(V, -1, dims=-2)
    AB = B - A                                                     # (Z, N, 2)
    AP = P[:, :, None, :] - A[:, None, :, :]                      # (Z, G, N, 2)
    denom = (AB * AB).sum(-1).clamp_min(1e-9)[:, None, :]        # (Z, 1, N)
    t = ((AP * AB[:, None]).sum(-1) / denom).clamp(0, 1)          # (Z, G, N)
    d = (AP - t[..., None] * AB[:, None]).norm(dim=-1)
    return d.min(dim=-1).values


def sample_polygon_b(V: torch.Tensor, per_edge: int):
    """V (Z, N, 2) -> punti (Z, N*per_edge, 2) e pesi d'arco (Z, N*per_edge)."""
    W = torch.roll(V, -1, dims=-2)
    t = (torch.arange(per_edge, device=V.device, dtype=V.dtype) + 0.5) / per_edge
    pts = V[:, :, None, :] * (1 - t)[None, None, :, None] + W[:, :, None, :] * t[None, None, :, None]
    w = (W - V).norm(dim=-1, keepdim=True).expand(-1, -1, per_edge) / per_edge
    return pts.flatten(1, 2), w.flatten(1, 2)


def contour_loss_b(V: torch.Tensor, G: torch.Tensor, per_edge: int = 8) -> torch.Tensor:
    """contour_loss_poly su Z coppie. V (Z, N, 2), G (Z, M, 2) allungati con pad_polygons. -> (Z,)"""
    q, w = sample_polygon_b(V, per_edge)
    a = (point_segment_distance_b(q, G) * w).sum(1) / w.sum(1).clamp_min(1e-9)
    g, wg = sample_polygon_b(G, 1)
    b = (point_segment_distance_b(g, V) * wg).sum(1) / wg.sum(1).clamp_min(1e-9)
    return 0.5 * (a + b)


def turning_angles_b(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    e_in = V - torch.roll(V, 1, dims=-2)
    e_out = torch.roll(V, -1, dims=-2) - V
    cross = e_in[..., 0] * e_out[..., 1] - e_in[..., 1] * e_out[..., 0]
    return torch.atan2(cross, (e_in * e_out).sum(-1) + eps)


def corner_penalty_b(V: torch.Tensor, p: float = 0.5, delta: float = 0.02) -> torch.Tensor:
    th = turning_angles_b(V)
    return ((th ** 2 + delta ** 2) ** (p / 2) - delta ** p).sum(-1)


def signed_area_b(V: torch.Tensor) -> torch.Tensor:
    x, y = V[..., 0], V[..., 1]
    return 0.5 * (x * torch.roll(y, -1, dims=-1) - torch.roll(x, -1, dims=-1) * y).sum(-1)


def area_loss_b(V: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    a_g = signed_area_b(G).abs().clamp_min(1.0)
    return (signed_area_b(V).abs() - a_g).abs() / a_g


def ordered_loss_b(V: torch.Tensor, Gs: torch.Tensor) -> torch.Tensor:
    """ordered_loss su Z coppie; Gs (Z, N, 2) è il contorno vero già ricampionato a N punti. -> (Z,)"""
    n = V.shape[1]
    idx = (torch.arange(n, device=V.device)[None, :] + torch.arange(n, device=V.device)[:, None]) % n   # (n, n)
    best = None
    for P in (Gs, torch.flip(Gs, dims=[1])):
        d = (V[:, None, :, :] - P[:, idx]).norm(dim=-1).mean(-1)     # (Z, n spostamenti)
        m = d.min(-1).values
        best = m if best is None else torch.minimum(best, m)
    return best


@torch.no_grad()
def prune_dev(V: torch.Tensor, max_dev: float = 0.5) -> torch.Tensor:
    """Potatura geometrica: toglie a uno a uno i punti la cui distanza dalla corda fra i due vicini è sotto
    max_dev (pixel), partendo dal più vicino alla corda. Togliere un punto sposta il contorno al massimo di
    quella distanza, quindi con max_dev sotto la precisione richiesta la potatura non cambia la metrica in modo
    apprezzabile. A differenza della soglia sull'angolo, non dipende dalla lunghezza dei lati: un tremolio di
    0.5 px su un lato di 20 px dà già un angolo di 0.025 rad, sopra la soglia di 0.02 di `prune`."""
    keep = V.clone()
    while len(keep) > 3:
        A, B = torch.roll(keep, 1, dims=0), torch.roll(keep, -1, dims=0)
        AB = B - A
        cross = (AB[:, 0] * (keep[:, 1] - A[:, 1]) - AB[:, 1] * (keep[:, 0] - A[:, 0])).abs()
        dev = cross / AB.norm(dim=1).clamp_min(1e-9)
        i = int(torch.argmin(dev))
        if dev[i] >= max_dev:
            break
        keep = torch.cat([keep[:i], keep[i + 1:]])
    return keep
