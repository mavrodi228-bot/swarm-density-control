from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import SwarmUNet
from swarm_core import gaussian_density, ring_density, make_obstacle, expert_flux, _normalise_density


class SyntheticSwarmDataset(Dataset):
    def __init__(self, samples=700, n=48, seed=1234):
        self.samples = samples
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx * 17)
        obstacle_name = rng.choice(["Нет", "Центральный блок", "Коридор", "Два блока", "Стена с проходом"], p=[.35,.22,.15,.15,.13])
        obstacle = make_obstacle(self.n, str(obstacle_name))

        k1 = int(rng.integers(1, 4))
        c1 = [(rng.uniform(.10,.90), rng.uniform(.10,.90)) for _ in range(k1)]
        s1 = [rng.uniform(.045,.14) for _ in range(k1)]
        rho = gaussian_density(self.n, c1, s1, obstacle=obstacle)

        if rng.random() < .25:
            target = ring_density(self.n, rng.uniform(.25,.78), rng.uniform(.25,.75), rng.uniform(.10,.25), rng.uniform(.035,.08), obstacle)
        else:
            k2 = int(rng.integers(1, 4))
            c2 = [(rng.uniform(.10,.90), rng.uniform(.10,.90)) for _ in range(k2)]
            s2 = [rng.uniform(.045,.14) for _ in range(k2)]
            target = gaussian_density(self.n, c2, s2, obstacle=obstacle)

        # Sometimes perturb the starting density to make the mapping less trivial.
        rho = _normalise_density(rho + rng.random((self.n,self.n))*1e-5, obstacle)
        jx, jy = expert_flux(rho, target, obstacle)
        scale = float(self.n*self.n)
        x = np.stack([rho*scale, target*scale, obstacle.astype(np.float32)], axis=0).astype(np.float32)
        y = np.stack([jx*scale, jy*scale], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--samples", type=int, default=700)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=str, default="model_data/swarm_unet.pt")
    ap.add_argument("--history", type=str, default="model_data/training_history.json")
    args = ap.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwarmUNet().to(device)
    ds = SyntheticSwarmDataset(args.samples, args.grid)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    hist = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            # Weighted MSE + small gradient consistency term.
            mse = torch.mean((pred-y)**2)
            dx_p = pred[:,:,:,1:] - pred[:,:,:,:-1]
            dx_y = y[:,:,:,1:] - y[:,:,:,:-1]
            dy_p = pred[:,:,1:,:] - pred[:,:,:-1,:]
            dy_y = y[:,:,1:,:] - y[:,:,:-1,:]
            grad = 0.5*(torch.mean((dx_p-dx_y)**2)+torch.mean((dy_p-dy_y)**2))
            loss = mse + 0.08*grad
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        mean_loss = float(np.mean(losses))
        hist.append({"epoch": epoch+1, "loss": mean_loss, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch+1:02d}/{args.epochs} loss={mean_loss:.6g}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    h = Path(args.history); h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
