"""Sequence models over chart-state windows: TCN / LSTM / Transformer.

Each model maps a [batch, window, features] candle sequence to a single logit
(P(trade is a winner), or any binary target). A shared fit_eval() trains on a
caller-provided train/test index split (the caller uses a group-by-symbol split
so no trade leaks across sides) and returns the test-set AUC.

Small on purpose -- the question is whether raw sequences beat engineered
features at all, not to squeeze a leaderboard. Research-only: no broker path.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                        # torch optional; script guards on this
    _TORCH = False


def _device(override: str = "auto"):
    if override and override != "auto":
        return torch.device(override)
    # MPS default OFF: dilated conv1d / LSTM kernels can deadlock on Metal,
    # especially after a prior run was killed mid-kernel. CPU is reliable here.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if _TORCH:
    class TCN(nn.Module):
        """Dilated temporal CNN -> global average pool -> logit."""
        def __init__(self, f, ch=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(f, ch, 3, padding=1, dilation=1), nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=8, dilation=8), nn.ReLU())
            self.head = nn.Linear(ch, 1)

        def forward(self, x):            # x: [B, W, F]
            z = self.net(x.transpose(1, 2))        # -> [B, ch, W]
            return self.head(z.mean(dim=2)).squeeze(1)

    class LSTMClf(nn.Module):
        def __init__(self, f, h=64):
            super().__init__()
            self.rnn = nn.LSTM(f, h, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(1)

    class TransformerClf(nn.Module):
        def __init__(self, f, d=64, heads=4, layers=2):
            super().__init__()
            self.proj = nn.Linear(f, d)
            enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1,
                                             batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            z = self.enc(self.proj(x))             # [B, W, d]
            return self.head(z.mean(dim=1)).squeeze(1)

    BUILDERS = {"tcn": TCN, "lstm": LSTMClf, "transformer": TransformerClf}


def fit_eval(X, y, train_idx, test_idx, model: str, epochs: int = 15,
             batch: int = 256, lr: float = 1e-3, seed: int = 0,
             device: str = "auto") -> float:
    """Train `model` on X[train_idx] -> return roc_auc on X[test_idx]."""
    from sklearn.metrics import roc_auc_score
    if not _TORCH:
        raise RuntimeError("torch not available")
    torch.manual_seed(seed)
    dev = _device(device)
    net = BUILDERS[model](X.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    pos = float(y[train_idx].mean())
    w = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=dev)  # class balance
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)

    Xtr = torch.tensor(X[train_idx], dtype=torch.float32)
    ytr = torch.tensor(y[train_idx], dtype=torch.float32)
    Xte = torch.tensor(X[test_idx], dtype=torch.float32).to(dev)
    n = len(Xtr)
    perm = np.random.default_rng(seed).permutation(n)
    net.train()
    for _ in range(epochs):
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb = Xtr[idx].to(dev); yb = ytr[idx].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xte)).cpu().numpy()
    return float(roc_auc_score(y[test_idx], p))


def _demo():
    if not _TORCH:
        print("seq_models._demo SKIP (torch unavailable)"); return
    # learnable toy: label = does the sequence trend up? sequence models must beat 0.5
    rng = np.random.default_rng(0)
    N, W, F = 240, 24, 16
    y = rng.integers(0, 2, N).astype(np.float32)
    X = rng.normal(0, 0.1, (N, W, F)).astype(np.float32)
    X[:, :, 0] += (y[:, None] * np.linspace(0, 1, W)[None, :]).astype(np.float32)
    tr, te = np.arange(0, 180), np.arange(180, N)
    auc = fit_eval(X, y, tr, te, "tcn", epochs=8)
    assert auc > 0.7, f"toy signal should be learnable, got {auc:.2f}"
    print(f"seq_models._demo OK  tcn toy AUC {auc:.2f} on {_device()}")


if __name__ == "__main__":
    _demo()
