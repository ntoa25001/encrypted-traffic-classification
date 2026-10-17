"""Classifiers working on the first-30-packets sequence (PPI) plus optional flow statistics."""
import torch
import torch.nn as nn


class StatsBranch(nn.Module):
    def __init__(self, stats_dim, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(stats_dim, out), nn.ReLU(), nn.Dropout(0.1))
        self.out = out

    def forward(self, s):
        return self.net(s)


class PpiCNN(nn.Module):
    """1D-CNN over the packet sequence (the CESNET multi-modal baseline in spirit)."""

    def __init__(self, n_classes, in_ch=3, stats_dim=0, width=64, use_stats=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width * 2, 3, padding=1), nn.BatchNorm1d(width * 2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width * 2, width * 2, 3, padding=1), nn.BatchNorm1d(width * 2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.stats = StatsBranch(stats_dim) if (use_stats and stats_dim > 0) else None
        d = width * 2 + (self.stats.out if self.stats else 0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, ppi, stats=None):
        h = self.conv(ppi).squeeze(-1)
        if self.stats is not None:
            h = torch.cat([h, self.stats(stats)], 1)
        return self.head(h)


class PpiLSTM(nn.Module):
    """Bidirectional LSTM over the packet sequence."""

    def __init__(self, n_classes, in_ch=3, stats_dim=0, hidden=96, use_stats=True):
        super().__init__()
        self.rnn = nn.LSTM(in_ch, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.stats = StatsBranch(stats_dim) if (use_stats and stats_dim > 0) else None
        d = hidden * 2 + (self.stats.out if self.stats else 0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, ppi, stats=None):
        out, _ = self.rnn(ppi.transpose(1, 2))
        h = out.mean(1)
        if self.stats is not None:
            h = torch.cat([h, self.stats(stats)], 1)
        return self.head(h)


class PpiTransformer(nn.Module):
    """Small Transformer encoder; each packet (IPT, DIR, SIZE) is one token.

    Supports self-supervised pre-training (masked packet reconstruction) via `reconstruct`.
    """

    def __init__(self, n_classes, in_ch=3, stats_dim=0, d_model=96, heads=4, layers=3, seq_len=30, use_stats=True):
        super().__init__()
        self.in_ch = in_ch
        self.embed = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.mask_tok, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.recon = nn.Linear(d_model, in_ch)
        self.stats = StatsBranch(stats_dim) if (use_stats and stats_dim > 0) else None
        d = d_model + (self.stats.out if self.stats else 0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def _encode(self, ppi, mask=None):
        x = self.embed(ppi.transpose(1, 2))                       # (B, L, D)
        if mask is not None:                                       # mask: (B, L) bool
            x = torch.where(mask.unsqueeze(-1), self.mask_tok.expand_as(x), x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1) + self.pos
        return self.norm(self.encoder(x))

    def reconstruct(self, ppi, mask):
        h = self._encode(ppi, mask)[:, 1:]
        return self.recon(h).transpose(1, 2)                       # (B, C, L)

    def forward(self, ppi, stats=None):
        h = self._encode(ppi)[:, 0]
        if self.stats is not None:
            h = torch.cat([h, self.stats(stats)], 1)
        return self.head(h)


def build_model(name, n_classes, in_ch, stats_dim, use_stats=True):
    cls = {"cnn": PpiCNN, "lstm": PpiLSTM, "transformer": PpiTransformer}[name]
    return cls(n_classes, in_ch=in_ch, stats_dim=stats_dim, use_stats=use_stats)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
