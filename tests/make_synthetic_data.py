#!/usr/bin/env python3
"""Generate a tiny synthetic tracking dataset to smoke-test the pipeline.

    python tests/make_synthetic_data.py --output /tmp/synth

Produces an OTB-style, a MOT-style and a raw-video sequence so that
``discover_sequences`` can be exercised on all three layouts.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def make_sequence(root: str, name: str, n: int = 90, w: int = 320, h: int = 240,
                  scale_drift: float = 0.0, fmt: str = "otb", seed: int = 0):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (h * 2, w * 2, 3), dtype=np.uint8), (9, 9), 0)
    tw, th = 48, 40
    target = cv2.GaussianBlur(rng.integers(0, 255, (th, tw, 3), dtype=np.uint8), (3, 3), 0)

    frames, gts = [], []
    for i in range(n):
        img = bg[i // 3: i // 3 + h, (i // 2) % w: (i // 2) % w + w].copy()
        s = 1.0 + scale_drift * i
        cw, ch = max(8, int(tw * s)), max(8, int(th * s))
        patch = cv2.resize(target, (cw, ch))
        x = min(max(int(40 + 1.8 * i + 10 * np.sin(i / 6.0)), 0), w - cw - 1)
        y = min(max(int(60 + 0.9 * i + 12 * np.cos(i / 5.0)), 0), h - ch - 1)
        img[y:y + ch, x:x + cw] = patch
        img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
        frames.append(img)
        gts.append([x, y, cw, ch])

    if fmt == "otb":
        d = os.path.join(root, "otb", name, "img")
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(f"{d}/{i + 1:04d}.jpg", f)
        with open(os.path.join(root, "otb", name, "groundtruth_rect.txt"), "w") as fh:
            for g in gts:
                fh.write("{},{},{},{}\n".format(*g))
    elif fmt == "mot":
        d = os.path.join(root, "mot", name, "img1")
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(root, "mot", name, "gt"), exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(f"{d}/{i + 1:06d}.jpg", f)
        with open(os.path.join(root, "mot", name, "gt", "gt.txt"), "w") as fh:
            for i, g in enumerate(gts):
                fh.write(f"{i + 1},1,{g[0]},{g[1]},{g[2]},{g[3]},1,1,1.0\n")
    elif fmt == "video":
        d = os.path.join(root, "video")
        os.makedirs(d, exist_ok=True)
        vw = cv2.VideoWriter(f"{d}/{name}.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        with open(f"{d}/{name}.txt", "w") as fh:
            for g in gts:
                fh.write("{} {} {} {}\n".format(*g))
    else:
        raise ValueError(fmt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/synth")
    args = p.parse_args()
    make_sequence(args.output, "scene_a", fmt="otb", seed=0)
    make_sequence(args.output, "scene_b", n=140, scale_drift=0.004, fmt="otb", seed=1)
    make_sequence(args.output, "scene_c", fmt="mot", seed=2)
    make_sequence(args.output, "scene_d", fmt="video", seed=3)
    print(f"synthetic dataset written to {args.output}")


if __name__ == "__main__":
    main()


