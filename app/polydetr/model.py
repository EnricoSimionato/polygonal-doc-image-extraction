"""PolyDETR: un DETR che emette, per ogni query, un punteggio e un poligono di N punti.

Struttura (vedi DESIGN-NOTES, passi 3-4):
1. backbone CNN preaddestrato (timm), feature a stride 16 e 32;
2. encoder: un transformer sulle feature a stride 32 (poche posizioni, contesto globale),
   poi fuse con quelle a stride 16 per avere più risoluzione nella cross-attention;
3. decoder: L strati di transformer con Q query apprese. Ogni query è un candidato «immagine»;
   la self-attention fra query evita i doppioni e aiuta a separare immagini adiacenti;
4. teste, dopo ogni strato del decoder:
   - un logit «c'è un'immagine» (sigmoide; niente classe esplicita «nessun oggetto»);
   - N punti (x, y) normalizzati in [0, 1]. Il primo strato li predice da zero, gli strati
     successivi predicono una correzione nello spazio logit (raffinamento iterativo, come i box
     di Deformable DETR), così ogni strato parte dal poligono del precedente.

Uscita di forward: dizionario con
    logits  (B, Q)            ultimo strato
    points  (B, Q, N, 2)      ultimo strato, in [0, 1] (x rispetto alla larghezza, y all'altezza)
    aux     lista di (logits, points) per gli strati intermedi, per le loss ausiliarie
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_pos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """Codifica posizionale sinusoidale 2D, forma (h*w, dim)."""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    y = (y.float() + 0.5) / h
    x = (x.float() + 0.5) / w
    d = dim // 4
    freq = torch.exp(torch.arange(d, device=device).float() * (-math.log(10000.0) / d))
    py, px = y.reshape(-1, 1) * freq * 2 * math.pi, x.reshape(-1, 1) * freq * 2 * math.pi
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], dim=1)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=3):
        super().__init__()
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class PolyDETR(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, d_model: int = 256,
                 n_queries: int = 20, n_points: int = 32, enc_layers: int = 1, dec_layers: int = 6,
                 n_heads: int = 8, ffn: int = 1024, dropout: float = 0.0):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(3, 4))  # stride 16 e 32
        c16, c32 = self.backbone.feature_info.channels()
        self.proj16 = nn.Conv2d(c16, d_model, 1)
        self.proj32 = nn.Conv2d(c32, d_model, 1)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.decoder_layers = nn.ModuleList(
            nn.TransformerDecoderLayer(d_model, n_heads, ffn, dropout, batch_first=True)
            for _ in range(dec_layers))
        self.query_content = nn.Embedding(n_queries, d_model)
        self.query_pos = nn.Embedding(n_queries, d_model)
        self.cls_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in range(dec_layers))
        self.pt_heads = nn.ModuleList(MLP(d_model, d_model, 2 * n_points) for _ in range(dec_layers))
        # centro iniziale del poligono di ogni query: funzione della sua posizione appresa
        self.center_head = nn.Linear(d_model, 2)
        self.n_points, self.d_model = n_points, d_model
        self._init_heads()

    def _init_heads(self):
        prior = 0.01  # probabilità iniziale bassa: quasi tutte le query sono vuote
        for h in self.cls_heads:
            nn.init.constant_(h.bias, -math.log((1 - prior) / prior))
        # gli strati di raffinamento partono da correzioni nulle
        for h in self.pt_heads[1:]:
            nn.init.zeros_(h.layers[-1].weight)
            nn.init.zeros_(h.layers[-1].bias)
        # il primo strato parte da un cerchio piccolo attorno a un centro che dipende dalla query
        nn.init.zeros_(self.pt_heads[0].layers[-1].weight)
        t = torch.arange(self.n_points).float() * 2 * math.pi / self.n_points
        circle = torch.stack([0.5 + 0.1 * t.cos(), 0.5 + 0.1 * t.sin()], 1).reshape(-1)
        self.pt_heads[0].layers[-1].bias.data = inverse_sigmoid(circle)
        # centri iniziali sparsi sulla pagina invece che tutti al centro
        nn.init.normal_(self.center_head.weight, std=0.1)
        nn.init.zeros_(self.center_head.bias)

    def train(self, mode: bool = True):
        """Le BatchNorm del backbone restano in valutazione: con batch piccoli le statistiche
        aggiornate sarebbero rumorose (le ricette DETR usano BatchNorm congelate)."""
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        f16, f32 = self.backbone((x - mean) / std)
        p32 = self.proj32(f32)
        h32, w32 = p32.shape[-2:]
        pos32 = sine_pos_2d(h32, w32, self.d_model, x.device)
        t32 = self.encoder(p32.flatten(2).transpose(1, 2) + pos32)          # (B, h32*w32, d)
        up = F.interpolate(t32.transpose(1, 2).reshape(B, self.d_model, h32, w32),
                           size=f16.shape[-2:], mode="bilinear", align_corners=False)
        m16 = self.proj16(f16) + up
        h16, w16 = m16.shape[-2:]
        memory = m16.flatten(2).transpose(1, 2) + sine_pos_2d(h16, w16, self.d_model, x.device)

        q = self.query_content.weight[None].expand(B, -1, -1)
        qpos = self.query_pos.weight[None].expand(B, -1, -1)
        outs = []
        ref = None
        for i, layer in enumerate(self.decoder_layers):
            q = layer(q + qpos, memory)
            logits = self.cls_heads[i](q).squeeze(-1)
            raw = self.pt_heads[i](q).view(B, -1, self.n_points, 2)
            if ref is None:
                # al primo strato il centro dipende dalla query tramite la posizione appresa
                raw = raw + self._center_offset(qpos)
                pts = raw.sigmoid()
            else:
                pts = (inverse_sigmoid(ref) + raw).sigmoid()
            outs.append((logits, pts))
            ref = pts.detach()  # come in Deformable DETR: il gradiente non attraversa i riferimenti
        logits, pts = outs[-1]
        return {"logits": logits, "points": pts, "aux": outs[:-1]}

    def _center_offset(self, qpos: torch.Tensor) -> torch.Tensor:
        return self.center_head(qpos)[:, :, None, :]
