"""Train and evaluate RGB-to-depth prediction with a shape-disjoint split."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .model import TinyDepth, load_model
else:
    from model import TinyDepth, load_model


class DepthDataset(Dataset):
    def __init__(self, folder, names, size):
        self.folder, self.names, self.size = folder, names, size
        for name in names:
            for kind in ("rgb", "depth", "mask"):
                if not (folder / kind / name).is_file():
                    raise ValueError(f"Missing {kind} for {name}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        rgb = cv2.imread(str(self.folder / "rgb" / name), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.folder / "depth" / name), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(self.folder / "mask" / name), cv2.IMREAD_GRAYSCALE)
        if (rgb is None or depth is None or mask is None or depth.dtype != np.uint16
                or depth.ndim != 2 or rgb.shape[:2] != depth.shape or mask.shape != depth.shape):
            raise ValueError(f"{name}: expected same-size RGB, uint16 depth and mask PNGs")
        size = (self.size, self.size)
        rgb = cv2.resize(rgb[..., ::-1], size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        valid = (cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0) & (depth > 0)
        if np.count_nonzero(valid) < 16:
            raise ValueError(f"{name}: fewer than 16 valid foreground depth pixels")
        return (torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255),
                torch.from_numpy(depth[None].astype(np.float32) / 1000),
                torch.from_numpy(valid[None]))


def dataset_splits(folder, seed):
    groups = {}
    for path in sorted((folder / "rgb").glob("*.png")):
        group, marker, view = path.stem.rpartition("_view_")
        if not marker or not group or not view.isdecimal():
            raise ValueError(f"{path.name}: expected SHAPE_view_NN.png")
        groups.setdefault(group, []).append(path.name)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Training needs at least three different mesh stems (train/validation/test).")
    random.Random(seed).shuffle(keys)
    n = max(1, len(keys) // 5)
    return {split: [name for key in selected for name in groups[key]]
            for split, selected in (("train", keys[2 * n:]), ("val", keys[n:2 * n]), ("test", keys[:n]))}


def loss_components(pred, truth, mask):
    log_residual = (pred.log() - truth.clamp_min(0.001).log()) * mask
    count = mask.sum(dim=(1, 2, 3)).clamp_min(1)
    silog = ((log_residual.square().sum(dim=(1, 2, 3)) / count
              - 0.85 * (log_residual.sum(dim=(1, 2, 3)) / count).square()).clamp_min(1e-8)).sqrt().mean()
    residual, valid = log_residual, mask
    terms = []
    for _ in range(3):
        for dim in (-1, -2):
            diff = residual.diff(dim=dim)
            pairs = (valid.narrow(dim, 0, valid.shape[dim] - 1)
                     & valid.narrow(dim, 1, valid.shape[dim] - 1))
            terms.append((diff.abs() * pairs).sum() / pairs.sum().clamp_min(1))
        residual = F.avg_pool2d(residual, 2)
        valid = F.avg_pool2d(valid.float(), 2) >= 1
    return silog, torch.stack(terms).mean()


@torch.inference_mode()
def evaluate(model, loader, device):
    stats = {"mae": [], "rmse": [], "abs_rel": [], "delta1": []}
    for rgb, truth, mask in loader:
        rgb, truth, mask = rgb.to(device), truth.to(device), mask.to(device)
        for p, t, m in zip(model(rgb), truth, mask):
            p, t = p[m], t[m]
            error = (p - t).abs()
            stats["mae"].append(error.mean().item())
            stats["rmse"].append(error.square().mean().sqrt().item())
            stats["abs_rel"].append((error / t).mean().item())
            stats["delta1"].append((torch.maximum(p / t, t / p) < 1.25).float().mean().item())
    return {key: float(np.mean(values)) for key, values in stats.items()}


def split_loader(args, names, training=False):
    return DataLoader(DepthDataset(args.data_dir, names, args.size), batch_size=args.batch_size,
                      shuffle=training, generator=torch.Generator().manual_seed(args.seed))


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available. Install a CUDA-enabled PyTorch build or use --device cpu.")
    if args.evaluate:
        model, state = load_model(args.evaluate, device)
        args.size = state["size"]
        metrics = evaluate(model, split_loader(args, state["splits"]["test"]), device)
        print(json.dumps({"test": metrics}, indent=2))
        return
    splits = dataset_splits(args.data_dir, args.seed)
    train, val, test = (split_loader(args, splits[k], training=(k == "train")) for k in ("train", "val", "test"))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "best_model.pth"
    model = TinyDepth().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for rgb, truth, mask in train:
            rgb, truth, mask = rgb.to(device), truth.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            a, b = loss_components(model(rgb), truth, mask)
            loss = a + args.grad_weight * b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(loss.item())
        scheduler.step()
        validation = evaluate(model.eval(), val, device)
        entry = {"epoch": epoch, "loss": float(np.mean(running)), "val": validation}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["mae"] < best:
            best = validation["mae"]
            torch.save({"architecture": "tinydepth-v1", "weights": model.state_dict(),
                        "size": args.size, "splits": splits, "epoch": epoch}, checkpoint)
    model, state = load_model(checkpoint, device)
    train_medians = [depth[mask].median().item() for _, depth, mask in train.dataset]
    constant = float(np.median(train_medians))
    record = {"device": device, "seed": args.seed, "size": args.size, "grad_weight": args.grad_weight,
              "parameters": sum(p.numel() for p in model.parameters()), "splits": splits,
              "best_epoch": state["epoch"], "history": history, "test": evaluate(model, test, device),
              "constant_depth": constant,
              "constant_test": evaluate(lambda rgb: torch.full_like(rgb[:, :1], constant), test, device)}
    (out / "training_history.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("best_epoch", "test", "constant_test")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=Path("out/depth"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--grad-weight", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluate", type=Path, help="Re-evaluate a saved checkpoint on its recorded test split")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.threads) < 1 or not 16 <= args.size <= 512
            or not np.isfinite(args.lr) or args.lr <= 0
            or not np.isfinite(args.grad_weight) or args.grad_weight < 0):
        parser.error("Require positive epochs, batch size, threads and learning rate, size 16-512, gradient weight >= 0.")
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
