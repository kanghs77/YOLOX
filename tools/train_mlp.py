#!/usr/bin/env python3
"""Stage 2 - train an MLP to predict CSRT parameters from frozen YOLOX-m features.

YOLOX-m is loaded from the official ``yolox_m.pth``, put in ``eval()`` mode and
fully frozen; only the MLP head is optimised.  The regression target is the
per-frame parameter vector written by ``tools/tune_csrt.py``.

Example
-------
    python tools/train_mlp.py \
        --labels outputs/csrt_labels/labels.jsonl \
        --yolox-ckpt weights/yolox_m.pth \
        --output outputs/mlp \
        --epochs 30 --batch-size 32 --cache-dir outputs/feat_cache
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csrt_mlp.features import FrozenYoloxFeatures  # noqa: E402
from csrt_mlp.model import CSRTParamMLP, build_loss  # noqa: E402
from csrt_mlp.params_spec import spec_from_json, spec_to_json  # noqa: E402
from csrt_mlp.train_utils import (  # noqa: E402
    CSRTLabelDataset,
    collate,
    group_split,
    read_labels,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the CSRT-parameter MLP")
    # data
    p.add_argument("--labels", required=True, help="labels.jsonl from stage 1")
    p.add_argument("--params-spec", default=None,
                   help="params_spec.json from stage 1 (default: next to --labels)")
    p.add_argument("--output", required=True)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--val-seqs", nargs="*", default=None,
                   help="explicit validation sequence names (overrides --val-ratio)")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="use every Nth labelled frame (consecutive frames are redundant)")
    p.add_argument("--drop-default-units", action="store_true",
                   help="skip frames whose unit kept the OpenCV defaults")
    p.add_argument("--weight-by-gain", action="store_true",
                   help="down-weight frames whose tuning gain was small (noisy labels)")
    # frozen backbone
    p.add_argument("--yolox-ckpt", required=True, help="path to yolox_m.pth")
    p.add_argument("--yolox-name", default="yolox-m",
                   choices=["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m",
                            "yolox-l", "yolox-x"])
    p.add_argument("--depth", type=float, default=None)
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--levels", nargs="+", default=["p3", "p4", "p5"],
                   choices=["p3", "p4", "p5"], help="FPN levels feeding the MLP")
    p.add_argument("--roi-size", type=int, default=3)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--no-context", action="store_true",
                   help="drop the globally pooled scene vector")
    p.add_argument("--no-geometry", action="store_true",
                   help="drop the box geometry features")
    p.add_argument("--cache-dir", default=None,
                   help="cache extracted features on disk (big speed-up over epochs)")
    # optimisation
    p.add_argument("--hidden", nargs="+", type=int, default=[1024, 512, 256])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "l1", "l2"])
    p.add_argument("--huber-beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=0, help="early stop (0 = off)")
    # runtime
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="fp16 feature extraction")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_features(batch: Dict[str, torch.Tensor], extractor, amp: bool,
                   device: torch.device) -> torch.Tensor:
    """Assemble the batch feature matrix from cached + freshly extracted rows."""
    n = batch["target"].shape[0]
    feats: List[torch.Tensor] = [None] * n  # type: ignore[list-item]

    if "feat" in batch:
        cached = batch["feat"].to(device, non_blocking=True).float()
        for row, pos in enumerate(batch["feat_pos"].tolist()):
            feats[pos] = cached[row]

    if "image" in batch:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            fresh = extractor(batch["image"], batch["box"])
        fresh = fresh.float()
        for row, pos in enumerate(batch["image_pos"].tolist()):
            feats[pos] = fresh[row]

    return torch.stack(feats)


def store_cache(batch: Dict[str, torch.Tensor], feats: torch.Tensor,
                dataset: CSRTLabelDataset):
    if dataset.cache.dir is None or "image_pos" not in batch:
        return
    arr = feats.detach().float().cpu().numpy()
    for pos in batch["image_pos"].tolist():
        rec = dataset.records[int(batch["index"][pos])]
        dataset.cache.put(dataset.cache_key(rec), arr[pos])


def lr_at(epoch: int, args) -> float:
    """Linear warm-up then cosine decay."""
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)


def run_epoch(loader, extractor, model, criterion, device, args, dataset,
              optimizer=None) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, total_mae, total_w, n = 0.0, None, 0.0, 0

    for batch in loader:
        target = batch["target"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)

        with torch.no_grad():
            feats = build_features(batch, extractor, args.amp, device)
        store_cache(batch, feats, dataset)

        with torch.set_grad_enabled(train):
            pred = model(feats)
            per_elem = criterion(pred, target)                 # (B, D)
            loss = (per_elem.mean(dim=1) * weight).sum() / weight.sum().clamp(min=1e-6)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        bs = target.shape[0]
        total_loss += float(loss.item()) * bs
        mae = (pred.detach() - target).abs().mean(dim=0).cpu().numpy()
        total_mae = mae * bs if total_mae is None else total_mae + mae * bs
        total_w += float(weight.sum().item())
        n += bs

    if n == 0:
        return {"loss": float("nan"), "mae": float("nan"), "per_param_mae": []}
    per_param = (total_mae / n).tolist()
    return {"loss": total_loss / n, "mae": float(np.mean(per_param)),
            "per_param_mae": per_param}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- labels -------------------------------------------------------
    spec_path = Path(args.params_spec) if args.params_spec else \
        Path(args.labels).parent / "params_spec.json"
    if not spec_path.is_file():
        raise SystemExit(f"params_spec.json not found at {spec_path}; pass --params-spec")
    space = spec_from_json(json.loads(spec_path.read_text()))

    records = read_labels(args.labels)
    if args.drop_default_units:
        records = [r for r in records if not r.get("used_default", False)]
    if args.frame_stride > 1:
        records = records[:: args.frame_stride]
    if not records:
        raise SystemExit("no training records left after filtering")

    train_recs, val_recs = group_split(records, args.val_ratio, args.seed, args.val_seqs)
    print(f"labels: {len(records)} frames | train {len(train_recs)} "
          f"({len({r['seq'] for r in train_recs})} seq) | "
          f"val {len(val_recs)} ({len({r['seq'] for r in val_recs})} seq)")
    if not train_recs:
        raise SystemExit("training split is empty - lower --val-ratio")

    # --- frozen backbone ---------------------------------------------
    print(f"loading frozen {args.yolox_name} from {args.yolox_ckpt} ...")
    extractor = FrozenYoloxFeatures(
        ckpt=args.yolox_ckpt,
        model_name=args.yolox_name,
        depth=args.depth,
        width=args.width,
        levels=args.levels,
        roi_size=args.roi_size,
        input_size=args.input_size,
        use_context=not args.no_context,
        use_geometry=not args.no_geometry,
        device=str(device),
    )
    n_frozen = sum(p.numel() for p in extractor.parameters())
    if extractor.load_report:
        print(f"  loaded {extractor.load_report['loaded']} tensors, "
              f"{len(extractor.load_report['missing'])} missing, "
              f"{len(extractor.load_report['unexpected'])} unexpected")
    print(f"  frozen parameters: {n_frozen/1e6:.2f}M | feature dim: {extractor.out_dim}")

    cache_sig = "|".join([
        args.yolox_name, str(args.depth), str(args.width), ",".join(args.levels),
        str(args.roi_size), str(args.input_size),
        str(not args.no_context), str(not args.no_geometry),
        Path(args.yolox_ckpt).name,
    ])

    def make_loader(recs, shuffle, weight_by_gain):
        ds = CSRTLabelDataset(recs, input_size=args.input_size,
                              cache_dir=args.cache_dir, cache_signature=cache_sig,
                              weight_by_gain=weight_by_gain)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=(device.type == "cuda"), drop_last=False,
                        persistent_workers=args.num_workers > 0)
        return ds, dl

    train_ds, train_loader = make_loader(train_recs, True, args.weight_by_gain)
    val_ds, val_loader = (make_loader(val_recs, False, False) if val_recs else (None, None))

    # --- model --------------------------------------------------------
    model = CSRTParamMLP(extractor.out_dim, len(space), hidden=args.hidden,
                         dropout=args.dropout).to(device)
    print(f"  MLP parameters   : {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(trainable) -> {len(space)} CSRT parameters")

    criterion = build_loss(args.loss, args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best = {"epoch": -1, "val": float("inf")}
    history = []
    bad_epochs = 0

    for epoch in range(args.epochs):
        lr = lr_at(epoch, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        tr = run_epoch(train_loader, extractor, model, criterion, device, args,
                       train_ds, optimizer)
        va = (run_epoch(val_loader, extractor, model, criterion, device, args, val_ds)
              if val_loader else {"loss": float("nan"), "mae": float("nan"),
                                  "per_param_mae": []})
        dt = time.time() - t0

        row = {"epoch": epoch, "lr": lr, "train_loss": tr["loss"], "train_mae": tr["mae"],
               "val_loss": va["loss"], "val_mae": va["mae"],
               "val_per_param_mae": va["per_param_mae"], "seconds": dt}
        history.append(row)
        print(f"epoch {epoch+1:3d}/{args.epochs} | lr {lr:.2e} | "
              f"train {tr['loss']:.5f} (mae {tr['mae']:.4f}) | "
              f"val {va['loss']:.5f} (mae {va['mae']:.4f}) | {dt:.1f}s", flush=True)

        monitor = va["loss"] if val_loader else tr["loss"]
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "params_spec": spec_to_json(space),
            "feature_config": {
                "yolox_name": args.yolox_name, "depth": args.depth, "width": args.width,
                "levels": args.levels, "roi_size": args.roi_size,
                "input_size": args.input_size, "use_context": not args.no_context,
                "use_geometry": not args.no_geometry, "in_dim": extractor.out_dim,
            },
            "mlp_config": {"hidden": args.hidden, "dropout": args.dropout,
                           "out_dim": len(space)},
            "val_loss": monitor,
        }
        torch.save(ckpt, out_dir / "last.pth")
        if monitor < best["val"]:
            best = {"epoch": epoch, "val": float(monitor)}
            torch.save(ckpt, out_dir / "best.pth")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"early stop: no improvement for {bad_epochs} epochs")
                break

    with open(out_dir / "history.json", "w") as fh:
        json.dump({"history": history, "best": best,
                   "param_names": [s.name for s in space]}, fh, indent=2)

    if history and history[-1]["val_per_param_mae"]:
        print("\nper-parameter validation MAE (normalised units):")
        for s, m in zip(space, history[-1]["val_per_param_mae"]):
            print(f"  {s.name:24s} {m:.4f}")
    print(f"\nbest epoch {best['epoch']+1} (loss {best['val']:.5f}) -> {out_dir/'best.pth'}")


if __name__ == "__main__":
    main()
