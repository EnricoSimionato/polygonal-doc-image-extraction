"""PolyDETR v2: le migliorie della famiglia DETR moderna, più un raffinamento del contorno che legge le
feature sui vertici. Motivazione, predizioni e criterio di abbandono in DESIGN-NOTES, passo 6.

Perché. La v1 sbaglia il bordo di circa 7 px d'ingresso (E9). La causa strutturale è che ogni strato del
decoder deve scrivere 2N = 64 coordinate a partire da un solo vettore di 256 numeri, ottenuto con
un'attention globale su una mappa a stride 16: nessuna feature del bordo arriva al punto che lo deve toccare.
La v2 cambia quattro cose, ognuna con un compito preciso:

1. **encoder multi-scala**, stride 8, 16 e 32. Default "hybrid", come l'encoder ibrido di RT-DETR
   (arXiv 2304.08069): attention solo sul livello a stride 32, poi fusione convoluzionale fra le scale.
   In alternativa "deform", l'encoder deformabile di Deformable DETR (arXiv 2010.04159) su quattro livelli,
   più lento su CPU a 512 x 384: 1.5 volte con ONNX Runtime e 2.6 con PyTorch (E10);
2. **query dall'encoder** (two-stage con selezione mista delle query, DINO, arXiv 2203.03605): ogni token
   propone un box e i Q migliori diventano i riferimenti iniziali. Il poligono di partenza di ogni query è
   il **rettangolo** del suo box, che per questo compito è già quasi la risposta;
3. **raffinamento sui vertici**, come Deep Snake (arXiv 2001.01629) ed E2EC (arXiv 2203.04074): dopo ogni
   strato del decoder si leggono le feature multi-scala, più una mappa a stride 4, esattamente nei N punti
   correnti del poligono. Una convoluzione circolare lungo il contorno ne ricava lo spostamento di ogni
   punto. Il punto che deve trovare il bordo vede finalmente il bordo;
4. **denoising** (DN-DETR, arXiv 2203.01305, con le query negative di DINO), in `build_dn_queries`:
   poligoni veri rumorosi entrano come query in più, con una maschera d'attention che impedisce di copiare.

Uscita di forward (stesso contratto della v1, più tre chiavi usate solo dal criterio):
    logits  (B, Q)          points (B, Q, N, 2) in [0, 1]      aux  [(logits, points)] degli strati intermedi
    enc     (logits (B, Q), boxes (B, Q, 4) x0 y0 x1 y1)        proposte dell'encoder scelte come query
    dn      [(logits, points)] per strato, solo in training con denoising     dn_meta  descrizione dei gruppi
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP, inverse_sigmoid


# ------------------------------------------------------------------ attention deformabile multi-scala

def ms_deform_attn(value, shapes, loc, weights):
    """Nucleo dell'attention deformabile in PyTorch puro (come la versione di riferimento di
    Deformable DETR, senza kernel CUDA, quindi gira anche su MPS e si esporta in ONNX).

    value   (B, S, Hh, Dh) con S = somma di h*w sui livelli
    shapes  lista di (h, w) per livello
    loc     (B, Lq, Hh, L, P, 2) posizioni di campionamento in [0, 1]
    weights (B, Lq, Hh, L, P) già normalizzati con softmax su L*P
    """
    B, S, Hh, Dh = value.shape
    _, Lq, _, L, P, _ = loc.shape
    vals = value.split([h * w for h, w in shapes], dim=1)
    grids = 2 * loc - 1
    sampled = []
    for lv, (h, w) in enumerate(shapes):
        v = vals[lv].flatten(2).transpose(1, 2).reshape(B * Hh, Dh, h, w)
        g = grids[:, :, :, lv].transpose(1, 2).flatten(0, 1)                     # (B*Hh, Lq, P, 2)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    aw = weights.transpose(1, 2).reshape(B * Hh, 1, Lq, L * P)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * aw).sum(-1)                # (B*Hh, Dh, Lq)
    return out.view(B, Hh * Dh, Lq).transpose(1, 2)


class MSDeformAttn(nn.Module):
    def __init__(self, d=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.d, self.L, self.Hh, self.P = d, n_levels, n_heads, n_points
        self.offsets = nn.Linear(d, n_heads * n_levels * n_points * 2)
        self.attn = nn.Linear(d, n_heads * n_levels * n_points)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        # inizializzazione di Deformable DETR: ogni testa guarda in una direzione diversa, a distanze 1..P
        nn.init.zeros_(self.offsets.weight)
        th = torch.arange(n_heads).float() * (2 * math.pi / n_heads)
        grid = torch.stack([th.cos(), th.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True).values).view(n_heads, 1, 1, 2).repeat(1, n_levels, n_points, 1)
        for i in range(n_points):
            grid[:, :, i] *= i + 1
        self.offsets.bias.data = grid.flatten()
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, query, ref, value, shapes):
        """query (B, Lq, d); ref (B, Lq, 2) centri oppure (B, Lq, 4) box cx cy w h, in [0, 1]."""
        B, Lq, _ = query.shape
        v = self.value(value).view(B, value.shape[1], self.Hh, self.d // self.Hh)
        off = self.offsets(query).view(B, Lq, self.Hh, self.L, self.P, 2)
        w = self.attn(query).view(B, Lq, self.Hh, self.L * self.P).softmax(-1).view(B, Lq, self.Hh, self.L, self.P)
        if ref.shape[-1] == 2:
            norm = torch.tensor([[w_, h_] for h_, w_ in shapes], dtype=query.dtype, device=query.device)
            loc = ref[:, :, None, None, None, :] + off / norm[None, None, None, :, None, :]
        else:  # box: gli spostamenti sono relativi alla taglia del box
            loc = ref[:, :, None, None, None, :2] + off / self.P * ref[:, :, None, None, None, 2:] * 0.5
        return self.out(ms_deform_attn(v, shapes, loc, w))


class EncoderLayer(nn.Module):
    def __init__(self, d, n_levels, n_heads, n_points, ffn, dropout):
        super().__init__()
        self.attn = MSDeformAttn(d, n_levels, n_heads, n_points)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, src, pos, ref, shapes):
        src = self.n1(src + self.drop(self.attn(src + pos, ref, src, shapes)))
        return self.n2(src + self.drop(self.ffn(src)))


class DecoderLayer(nn.Module):
    def __init__(self, d, n_levels, n_heads, n_points, ffn, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross = MSDeformAttn(d, n_levels, n_heads, n_points)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, qpos, ref_box, memory, shapes, attn_mask=None):
        k = q + qpos
        q = self.n1(q + self.drop(self.self_attn(k, k, q, attn_mask=attn_mask, need_weights=False)[0]))
        q = self.n2(q + self.drop(self.cross(q + qpos, ref_box, memory, shapes)))
        return self.n3(q + self.drop(self.ffn(q)))


class ConvBlock(nn.Module):
    """1x1 (c_in -> d), 3x3 depthwise, 1x1 (d -> d), con GroupNorm e ReLU: leggero anche a stride 8."""

    def __init__(self, c_in, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, d, 1), nn.GroupNorm(32, d), nn.ReLU(),
                                 nn.Conv2d(d, d, 3, padding=1, groups=d), nn.GroupNorm(32, d), nn.ReLU(),
                                 nn.Conv2d(d, d, 1), nn.GroupNorm(32, d))
        self.skip = nn.Conv2d(c_in, d, 1) if c_in != d else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.skip(x))


class HybridEncoder(nn.Module):
    """Encoder ibrido alla RT-DETR: attention solo sul livello a stride 32 (pochi token, contesto globale),
    poi fusione convoluzionale fra le scale, dall'alto in basso (FPN) e dal basso in alto (PAN).
    A 512 x 384 l'attention vede 192 token invece dei 4080 dell'encoder deformabile su tutti i livelli."""

    def __init__(self, d, n_heads, ffn, dropout, layers=1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d, n_heads, ffn, dropout, batch_first=True)
        self.attn = nn.TransformerEncoder(enc, layers)
        self.td16, self.td8 = ConvBlock(2 * d, d), ConvBlock(2 * d, d)
        self.down8, self.down16 = (nn.Conv2d(d, d, 3, stride=2, padding=1, groups=d),
                                   nn.Conv2d(d, d, 3, stride=2, padding=1, groups=d))
        self.bu16, self.bu32 = ConvBlock(2 * d, d), ConvBlock(2 * d, d)

    def forward(self, s8, s16, s32):
        from .model import sine_pos_2d

        B, d, h, w = s32.shape
        t = self.attn(s32.flatten(2).transpose(1, 2) + sine_pos_2d(h, w, d, s32.device).to(s32.dtype))
        s32 = t.transpose(1, 2).reshape(B, d, h, w)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="nearest")
        p16 = self.td16(torch.cat([up(s32, s16), s16], 1))
        p8 = self.td8(torch.cat([up(p16, s8), s8], 1))
        n16 = self.bu16(torch.cat([up(self.down8(p8), p16), p16], 1))
        n32 = self.bu32(torch.cat([up(self.down16(n16), s32), s32], 1))
        return [p8, n16, n32]


# ------------------------------------------------------------------ geometria delle query

def box_sine_embed(box: torch.Tensor, d: int) -> torch.Tensor:
    """Codifica sinusoidale di un box (cx, cy, w, h) in [0, 1]: d/4 frequenze per coordinata."""
    k = d // 4
    freq = 10000 ** (2 * (torch.arange(k, device=box.device, dtype=box.dtype) // 2) / k)
    x = box[..., None] * 2 * math.pi / freq                                    # (..., 4, k)
    x = torch.stack([x[..., 0::2].sin(), x[..., 1::2].cos()], -1).flatten(-2)
    return x.flatten(-2)


def xyxy_to_cxcywh(b):
    return torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                        b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)


def cxcywh_to_xyxy(b):
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], -1)


def points_to_box(pts: torch.Tensor) -> torch.Tensor:
    """(..., N, 2) -> (..., 4) cx cy w h, con una taglia minima per non degenerare."""
    lo, hi = pts.min(-2).values, pts.max(-2).values
    return torch.cat([(lo + hi) / 2, (hi - lo).clamp_min(1e-3)], -1)


def rect_points(box_xyxy: torch.Tensor, n: int) -> torch.Tensor:
    """n punti sul perimetro del rettangolo, n/4 per lato a partire da ogni angolo, in senso orario
    sull'immagine (x a destra, y in basso) dall'angolo in alto a sinistra. I quattro angoli sono sempre
    fra i punti: con punti equispaziati sull'arco due angoli andrebbero persi e il poligono iniziale
    taglierebbe gli angoli. Il verso è indifferente per le loss (area_loss e ordered_loss lo ignorano).
    (..., 4) -> (..., n, 2); richiede n multiplo di 4."""
    assert n % 4 == 0, "il numero di punti deve essere multiplo di 4"
    x0, y0, x1, y1 = box_xyxy.unbind(-1)
    corners = torch.stack([torch.stack([x0, y0], -1), torch.stack([x1, y0], -1),
                           torch.stack([x1, y1], -1), torch.stack([x0, y1], -1)], -2)       # (..., 4, 2)
    nxt = torch.roll(corners, -1, dims=-2)
    t = torch.arange(n // 4, device=box_xyxy.device, dtype=box_xyxy.dtype) / (n // 4)          # (n/4,)
    pts = corners[..., :, None, :] + (nxt - corners)[..., :, None, :] * t[:, None]            # (..., 4, n/4, 2)
    return pts.flatten(-3, -2)


# ------------------------------------------------------------------ raffinamento sui vertici

class CircConv1d(nn.Module):
    """Convoluzione 1D circolare lungo il contorno, scritta come gather più prodotto di matrici.
    Equivale a nn.Conv1d(padding_mode="circular") con dilatazione, ma su CPU la versione di PyTorch con
    dilatazione passa da un'implementazione lenta (26 ms a chiamata a 512 x 384, misurato), e questa no."""

    def __init__(self, c_in, c_out, k, dilation, n_points):
        super().__init__()
        self.lin = nn.Linear(c_in * k, c_out)
        off = (torch.arange(k) - k // 2) * dilation
        self.register_buffer("idx", (torch.arange(n_points)[:, None] + off[None, :]) % n_points, persistent=False)

    def forward(self, h):
        """h (Z, N, c_in) -> (Z, N, c_out)"""
        Z, N, C = h.shape
        return self.lin(h[:, self.idx].reshape(Z, N, -1))


class VertexRefiner(nn.Module):
    """Legge le feature nei punti del poligono e ne ricava lo spostamento di ogni punto.

    Le feature vengono da tutti i livelli della memoria (d canali, sommati) e dalla mappa fine a stride 4
    (d_fine canali). A queste si somma la query (la forma globale) e un embedding dell'indice del punto.
    Tre convoluzioni circolari lungo il contorno (kernel 9, dilatazioni 1, 2, 4) danno a ogni punto un
    campo recettivo di circa 29 vicini, abbastanza per capire dove sta l'angolo di un rettangolo con N = 32.
    Uscite: lo spostamento in spazio logit (N, 2) e un riassunto (d) che torna nella query.
    """

    def __init__(self, d, d_fine, n_points, hidden=128):
        super().__init__()
        self.inp = nn.Linear(d + d_fine, hidden)
        self.q = nn.Linear(d, hidden)
        self.idx = nn.Parameter(torch.randn(n_points, hidden) * 0.02)
        self.convs = nn.ModuleList(CircConv1d(hidden, hidden, 9, dil, n_points) for dil in (1, 2, 4))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))
        self.delta = nn.Linear(hidden, 2)
        self.back = nn.Linear(hidden, d)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.back.weight)
        nn.init.zeros_(self.back.bias)

    def forward(self, q, pts, maps, fine):
        """q (B, Q, d); pts (B, Q, N, 2) in [0, 1]; maps lista di (B, d, h, w); fine (B, d_fine, h, w)."""
        B, Q, N, _ = pts.shape
        g = (2 * pts - 1).view(B, Q * N, 1, 2)
        f = sum(F.grid_sample(m, g, mode="bilinear", padding_mode="border", align_corners=False) for m in maps)
        ff = F.grid_sample(fine, g, mode="bilinear", padding_mode="border", align_corners=False)
        f = torch.cat([f, ff], 1).view(B, -1, Q, N).permute(0, 2, 3, 1)               # (B, Q, N, d + d_fine)
        h = self.inp(f) + self.q(q)[:, :, None] + self.idx
        h = h.reshape(B * Q, N, -1)                                                    # (B*Q, N, hidden)
        for conv, nrm in zip(self.convs, self.norms):
            h = h + F.relu(nrm(conv(h)))
        h = h.reshape(B, Q, N, -1)
        return self.delta(h), self.back(h.mean(2))


# ------------------------------------------------------------------ modello

class PolyDETRv2(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, d_model: int = 256,
                 n_queries: int = 40, n_points: int = 32, enc_layers: int = 0, dec_layers: int = 6,
                 n_heads: int = 8, n_levels: int = 4, n_sample: int = 4, ffn: int = 1024, dropout: float = 0.0,
                 d_fine: int = 64, encoder: str = "hybrid"):
        """encoder: "hybrid" (default, alla RT-DETR, 3 livelli: stride 8, 16, 32) oppure "deform" (Deformable
        DETR, n_levels livelli fino a stride 64). enc_layers = 0 sceglie 1 per hybrid e 3 per deform."""
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(1, 2, 3, 4))              # stride 4, 8, 16, 32
        c4, c8, c16, c32 = self.backbone.feature_info.channels()
        gn = lambda c: nn.GroupNorm(32, c)
        self.encoder_type = encoder
        if encoder == "hybrid":
            n_levels = 3
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, d_model, 1), gn(d_model)) for c in (c8, c16, c32)])
        if n_levels == 4:
            self.proj.append(nn.Sequential(nn.Conv2d(c32, d_model, 3, stride=2, padding=1), gn(d_model)))
        self.fine = nn.Sequential(nn.Conv2d(c4, d_fine, 3, padding=1), nn.GroupNorm(8, d_fine), nn.ReLU())
        if encoder == "hybrid":
            self.encoder = HybridEncoder(d_model, n_heads, ffn, dropout, layers=enc_layers or 1)
        else:
            self.level_embed = nn.Parameter(torch.randn(n_levels, d_model) * 0.02)
            self.encoder = nn.ModuleList(EncoderLayer(d_model, n_levels, n_heads, n_sample, ffn, dropout)
                                         for _ in range(enc_layers or 3))
        # two-stage: ogni token dell'encoder propone un box
        self.enc_out = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.enc_cls = nn.Linear(d_model, 1)
        self.enc_box = MLP(d_model, d_model, 4)
        # decoder
        self.decoder = nn.ModuleList(DecoderLayer(d_model, n_levels, n_heads, n_sample, ffn, dropout)
                                     for _ in range(dec_layers))
        self.query_content = nn.Embedding(n_queries, d_model)    # contenuto appreso (selezione mista di DINO)
        self.dn_content = nn.Embedding(2, d_model)               # contenuto delle query di denoising: 0 vera, 1 negativa
        self.pos_mlp = MLP(d_model, d_model, d_model, 2)          # posizione della query dal suo box
        self.cls_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in range(dec_layers))
        self.refiners = nn.ModuleList(VertexRefiner(d_model, d_fine, n_points) for _ in range(dec_layers))
        self.n_queries, self.n_points, self.d_model, self.n_levels = n_queries, n_points, d_model, n_levels
        # NMS sulle proposte dell'encoder prima di scegliere le query (0 = spenta): un'immagine copre centinaia di
        # token, e senza NMS le Q query migliori nascono in gruppo sulle stesse immagini (mediana 6 per immagine nel
        # ciclo 1). Dense Distinct Query (arXiv 2303.12776) sceglie query distinte per lo stesso motivo.
        self.proposal_nms, self.proposal_pool = 0.0, 300
        prior = -math.log((1 - 0.01) / 0.01)
        for h in list(self.cls_heads) + [self.enc_cls]:
            nn.init.constant_(h.bias, prior)
        nn.init.zeros_(self.enc_box.layers[-1].weight)
        nn.init.zeros_(self.enc_box.layers[-1].bias)

    def train(self, mode: bool = True):
        """BatchNorm del backbone congelate, come nella v1."""
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        return self

    # -------------------------------------------------------------- pezzi

    def _features(self, x):
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        f4, f8, f16, f32 = self.backbone((x - mean) / std)
        srcs = [self.proj[0](f8), self.proj[1](f16), self.proj[2](f32)]
        if self.n_levels == 4:
            srcs.append(self.proj[3](f32))
        return srcs, self.fine(f4)

    def _encode(self, srcs):
        from .model import sine_pos_2d

        B = srcs[0].shape[0]
        if self.encoder_type == "hybrid":
            srcs = self.encoder(*srcs)
        shapes = [tuple(s.shape[-2:]) for s in srcs]
        flat, pos, ref, props = [], [], [], []
        for lv, s in enumerate(srcs):
            h, w = s.shape[-2:]
            flat.append(s.flatten(2).transpose(1, 2))
            if self.encoder_type != "hybrid":
                pos.append(sine_pos_2d(h, w, self.d_model, s.device).to(s.dtype) + self.level_embed[lv])
            ys, xs = torch.meshgrid(torch.arange(h, device=s.device), torch.arange(w, device=s.device), indexing="ij")
            c = torch.stack([(xs.float() + 0.5) / w, (ys.float() + 0.5) / h], -1).reshape(-1, 2).to(s.dtype)
            ref.append(c)
            # proposta di base del token: un box centrato su di lui, di lato 0.05 * 2^livello
            props.append(torch.cat([c, torch.full_like(c, 0.05 * 2 ** lv)], -1))
        src = torch.cat(flat, 1)
        if self.encoder_type != "hybrid":
            pos = torch.cat(pos, 0)[None]
            ref = torch.cat(ref, 0)[None].expand(B, -1, -1)
            for layer in self.encoder:
                src = layer(src, pos, ref, shapes)
        props = torch.cat(props, 0)[None].expand(B, -1, -1)
        return src, shapes, props

    def _distinct_topk(self, logits, boxes, Q):
        """Per ogni pagina: le proposal_pool proposte migliori, NMS sui box con soglia proposal_nms, poi le prime Q.
        Se dopo la NMS ne restano meno di Q si completa con le migliori scartate."""
        from torchvision.ops import nms

        pool = logits.topk(min(self.proposal_pool, logits.shape[1]), dim=1).indices
        out = []
        for b in range(logits.shape[0]):
            idx = pool[b]
            bx = cxcywh_to_xyxy(boxes[b, idx]).float()
            keep = nms(bx, logits[b, idx].float(), self.proposal_nms)
            if len(keep) < Q:
                rest = torch.ones(len(idx), dtype=torch.bool, device=idx.device)
                rest[keep] = False
                keep = torch.cat([keep, torch.nonzero(rest).flatten()[: Q - len(keep)]])
            out.append(idx[keep[:Q]])
        return torch.stack(out)

    def _maps(self, memory, shapes):
        out, i = [], 0
        for h, w in shapes:
            out.append(memory[:, i:i + h * w].transpose(1, 2).reshape(memory.shape[0], -1, h, w))
            i += h * w
        return out

    # -------------------------------------------------------------- forward

    def forward(self, x, dn=None):
        """dn: dizionario di build_dn_queries (solo in training), oppure None."""
        srcs, fine = self._features(x)
        memory, shapes, props = self._encode(srcs)
        B = x.shape[0]
        Q, N = self.n_queries, self.n_points

        # two-stage: punteggio e box per ogni token, si tengono i Q migliori
        eo = self.enc_out(memory)
        enc_logits = self.enc_cls(eo).squeeze(-1)                                       # (B, S)
        enc_boxes = (inverse_sigmoid(props) + self.enc_box(eo)).sigmoid()               # (B, S, 4) cx cy w h
        if self.proposal_nms > 0:
            top = self._distinct_topk(enc_logits, enc_boxes, Q)
        else:
            top = enc_logits.topk(Q, dim=1).indices
        sel_logits = enc_logits.gather(1, top)
        sel_boxes = enc_boxes.gather(1, top[..., None].expand(-1, -1, 4))
        enc = (sel_logits, cxcywh_to_xyxy(sel_boxes).clamp(0, 1))

        ref_box = sel_boxes.detach()
        pts = rect_points(cxcywh_to_xyxy(ref_box).clamp(0, 1), N)                     # (B, Q, N, 2)
        q = self.query_content.weight[None].expand(B, -1, -1)
        mask = None
        n_dn = 0
        if dn is not None:
            n_dn = dn["points"].shape[1]
            q = torch.cat([self.dn_content(dn["kind"]), q], 1)
            pts = torch.cat([dn["points"].to(pts.dtype), pts], 1)
            mask = dn["attn_mask"]

        maps = self._maps(memory, shapes)
        outs = []
        for i, layer in enumerate(self.decoder):
            box = points_to_box(pts)
            qpos = self.pos_mlp(box_sine_embed(box, self.d_model))
            q = layer(q, qpos, box, memory, shapes, attn_mask=mask)
            delta, back = self.refiners[i](q, pts, maps, fine)
            q = q + back
            new = (inverse_sigmoid(pts) + delta).sigmoid()
            outs.append((self.cls_heads[i](q).squeeze(-1), new))
            pts = new.detach()          # come i box di Deformable DETR: il gradiente non attraversa i riferimenti
        main = [(lg[:, n_dn:], p[:, n_dn:]) for lg, p in outs]
        out = {"logits": main[-1][0], "points": main[-1][1], "aux": main[:-1], "enc": enc}
        if dn is not None:
            out["dn"] = [(lg[:, :n_dn], p[:, :n_dn]) for lg, p in outs]
            out["dn_meta"] = dn
        return out


# ------------------------------------------------------------------ denoising

@torch.no_grad()
def build_dn_queries(targets, n_points: int, n_dn: int = 60, box_noise: float = 0.4, n_queries: int = 40,
                     device="cpu"):
    """Query di denoising per un batch (DN-DETR con le negative di DINO).

    Per ogni immagine vera: il contorno ricampionato a N punti, poi un rumore di box (centro spostato fino a
    box_noise * metà lato, lati scalati di un fattore in [1 - box_noise, 1 + box_noise]) e un tremolio dei
    punti pari al 2% del lato. La copia **positiva** ha rumore fino a box_noise e deve tornare al contorno
    vero; la copia **negativa** ha rumore fra box_noise e 2 * box_noise e deve dire «nessuna immagine».
    Gruppi: G = n_dn // (2 * max_gt), ognuno con una copia positiva e una negativa di ogni immagine vera.

    Maschera d'attention (True = vietato): le query normali non vedono le query di denoising, e i gruppi di
    denoising non si vedono fra loro; altrimenti la risposta passerebbe dalle copie alle query normali.
    Restituisce None se il batch non ha immagini vere.
    """
    from .losses import resample_closed

    B = len(targets)
    K = max(len(t["polys"]) for t in targets)
    if K == 0:
        return None
    G = max(1, n_dn // (2 * K))
    M = G * 2 * K
    pts = torch.full((B, M, n_points, 2), 0.5)
    kind = torch.ones(B, M, dtype=torch.long)
    gt_idx = torch.full((B, M), -1, dtype=torch.long)       # immagine vera di ogni query positiva
    valid = torch.zeros(B, M, dtype=torch.bool)
    for b, t in enumerate(targets):
        H, W = t["size"]
        scale = torch.tensor([W, H], dtype=torch.float32)
        for k, P in enumerate(t["polys"]):
            base = resample_closed(P.float(), n_points) / scale                        # (N, 2) in [0, 1]
            lo, hi = base.min(0).values, base.max(0).values
            c, wh = (lo + hi) / 2, (hi - lo).clamp_min(1e-3)
            for g in range(G):
                for neg in (0, 1):
                    j = g * 2 * K + neg * K + k
                    mag = box_noise * (1 + torch.rand(1)) if neg else box_noise * torch.rand(1)
                    sgn = torch.randint(0, 2, (2,)) * 2 - 1
                    shift = sgn * mag * torch.rand(2) * wh / 2
                    s = 1 + (torch.rand(2) * 2 - 1) * mag
                    jit = torch.randn(n_points, 2) * 0.02 * wh
                    pts[b, j] = ((base - c) * s + c + shift + jit).clamp(1e-3, 1 - 1e-3)
                    kind[b, j] = neg
                    gt_idx[b, j] = -1 if neg else k
                    valid[b, j] = True
    T = M + n_queries
    mask = torch.zeros(T, T, dtype=torch.bool)
    mask[M:, :M] = True
    for g in range(G):
        a, z = g * 2 * K, (g + 1) * 2 * K
        mask[a:z, :a] = True
        mask[a:z, z:M] = True
    return {"points": pts.to(device), "kind": kind.to(device), "gt_idx": gt_idx, "valid": valid,
            "attn_mask": mask.to(device), "groups": G, "max_gt": K}


def build_model(args: dict):
    """Costruisce la v1 o la v2 dagli argomenti salvati nel checkpoint (le v1 non hanno --arch)."""
    if args.get("arch", "v1") == "v2":
        m = PolyDETRv2(args["backbone"], pretrained=False, n_queries=args["queries"], n_points=args["points"],
                       enc_layers=args.get("enc_layers", 0), dec_layers=args["dec_layers"],
                       encoder=args.get("encoder", "hybrid"))
        m.proposal_nms = args.get("proposal_nms", 0.0)
        return m
    from .model import PolyDETR

    return PolyDETR(args["backbone"], pretrained=False, n_queries=args["queries"], n_points=args["points"],
                    dec_layers=args["dec_layers"])
