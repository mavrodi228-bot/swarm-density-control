from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class SwarmUNet(nn.Module):
    """Small fully-convolutional network: [rho,target,obstacle] -> [Jx,Jy]."""
    def __init__(self, base=16):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base*2)
        self.b = ConvBlock(base*2, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.b(F.avg_pool2d(b, 2))
        u = self.u2(c)
        if u.shape[-2:] != b.shape[-2:]:
            u = F.interpolate(u, size=b.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d2(torch.cat([u, b], dim=1))
        u = self.u1(u)
        if u.shape[-2:] != a.shape[-2:]:
            u = F.interpolate(u, size=a.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d1(torch.cat([u, a], dim=1))
        return self.out(u)


def load_model(weights_path: str | Path, device="cpu"):
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"SwarmUNet weights not found: {path}")
    model = SwarmUNet()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Original distributed files contain the state dict directly. Laboratory
    # checkpoints also retain optimizer/scheduler state for resumable training.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_flux(model, rho: np.ndarray, target: np.ndarray, obstacle: np.ndarray, device="cpu"):
    n = rho.shape[0]
    scale = float(n*n)
    x = np.stack([rho*scale, target*scale, obstacle.astype(np.float32)], axis=0)[None]
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32)).to(device)
        y = model(xt).cpu().numpy()[0] / scale
    jx, jy = y[0], y[1]
    free = (~obstacle).astype(np.float64)
    return jx*free, jy*free
