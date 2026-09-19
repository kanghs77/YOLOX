"""Dataset / splitting / feature-cache helpers for MLP training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import preprocess_image

__all__ = ["LabelRecord", "CSRTLabelDataset", "read_labels", "group_split", "collate"]


def read_labels(path: str) -> List[dict]:
    """Read the JSONL produced by ``tools/tune_csrt.py``."""
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} contains no labels")
    return records


def group_split(
    records: Sequence[dict],
    val_ratio: float = 0.2,
    seed: int = 0,
    val_sequences: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Split by *sequence* so no frame of a val clip is ever seen in training."""
    names = sorted({r["seq"] for r in records})
    if val_sequences:
        val_names = {n for n in names if n in set(val_sequences)}
        missing = set(val_sequences) - val_names
        if missing:
            raise ValueError(f"--val-seqs not found in labels: {sorted(missing)}")
    else:
        rng = np.random.default_rng(seed)
        order = list(names)
        rng.shuffle(order)
        n_val = max(1, int(round(len(order) * val_ratio))) if len(order) > 1 else 0
        val_names = set(order[:n_val])

    train = [r for r in records if r["seq"] not in val_names]
    val = [r for r in records if r["seq"] in val_names]
    return train, val


class _FeatureCache:
    """Tiny on-disk float16 cache keyed by (image, box, extractor config)."""

    def __init__(self, cache_dir: Optional[str], signature: str):
        self.dir = Path(cache_dir) if cache_dir else None
        self.signature = signature
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.md5((self.signature + "|" + key).encode()).hexdigest()
        return self.dir / digest[:2] / f"{digest}.npy"

    def get(self, key: str) -> Optional[np.ndarray]:
        if not self.dir:
            return None
        p = self._path(key)
        if p.is_file():
            try:
                return np.load(p)
            except Exception:
                return None
        return None

    def put(self, key: str, value: np.ndarray):
        if not self.dir:
            return
        p = self._path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, value.astype(np.float16))
            os.replace(tmp, p)
        except Exception:
            pass


class CSRTLabelDataset(Dataset):
    """One item = one annotated frame.

    Yields the preprocessed image, the tracked box in input coordinates and the
    normalised CSRT parameter vector.  Features are computed on GPU in the
    training loop (batched) rather than here, unless a cache hit is available.
    """

    def __init__(
        self,
        records: Sequence[dict],
        input_size: int = 640,
        cache_dir: Optional[str] = None,
        cache_signature: str = "",
        weight_by_gain: bool = False,
        min_gain_weight: float = 0.25,
    ):
        self.records = list(records)
        self.input_size = int(input_size)
        self.cache = _FeatureCache(cache_dir, cache_signature)
        self.weight_by_gain = bool(weight_by_gain)
        self.min_gain_weight = float(min_gain_weight)

    def __len__(self) -> int:
        return len(self.records)

    # -- helpers -----------------------------------------------------------
    def _weight(self, rec: dict) -> float:
        if not self.weight_by_gain:
            return 1.0
        gain = float(rec.get("gain", rec.get("score", 0.0) - rec.get("baseline_score", 0.0)))
        # a unit where tuning barely helped carries a noisy label -> down-weight
        return float(np.clip(self.min_gain_weight + 4.0 * max(gain, 0.0), self.min_gain_weight, 1.0))

    def _read_image(self, rec: dict) -> np.ndarray:
        import cv2

        if rec.get("image"):
            img = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"cannot read image {rec['image']}")
            return img
        cap = cv2.VideoCapture(rec["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(rec["frame_no"]))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read frame {rec['frame_no']} of {rec['video']}")
        return img

    def cache_key(self, rec: dict) -> str:
        src = rec.get("image") or f"{rec.get('video')}:{rec.get('frame_no')}"
        b = ",".join(f"{float(v):.2f}" for v in rec["bbox"])
        return f"{src}|{b}"

    # -- item --------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index]
        target = torch.tensor(rec["param_vec"], dtype=torch.float32)
        weight = torch.tensor(self._weight(rec), dtype=torch.float32)

        key = self.cache_key(rec)
        cached = self.cache.get(key)
        if cached is not None:
            return {
                "feat": torch.from_numpy(cached.astype(np.float32)),
                "target": target,
                "weight": weight,
                "index": torch.tensor(index, dtype=torch.long),
                "cached": torch.tensor(1, dtype=torch.bool),
            }

        img = self._read_image(rec)
        chw, r = preprocess_image(img, self.input_size)
        x, y, w, h = (float(v) for v in rec["bbox"])
        box = torch.tensor([x * r, y * r, (x + w) * r, (y + h) * r], dtype=torch.float32)
        return {
            "image": torch.from_numpy(chw),
            "box": box,
            "target": target,
            "weight": weight,
            "index": torch.tensor(index, dtype=torch.long),
            "cached": torch.tensor(0, dtype=torch.bool),
        }


def collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Keeps cached and uncached items apart so both paths stay batched."""
    out: Dict[str, torch.Tensor] = {
        "target": torch.stack([b["target"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
        "index": torch.stack([b["index"] for b in batch]),
        "cached": torch.stack([b["cached"] for b in batch]),
    }
    cached = [b for b in batch if bool(b["cached"])]
    fresh = [b for b in batch if not bool(b["cached"])]
    if cached:
        out["feat"] = torch.stack([b["feat"] for b in cached])
        out["feat_pos"] = torch.tensor(
            [i for i, b in enumerate(batch) if bool(b["cached"])], dtype=torch.long
        )
    if fresh:
        out["image"] = torch.stack([b["image"] for b in fresh])
        out["box"] = torch.stack([b["box"] for b in fresh])
        out["image_pos"] = torch.tensor(
            [i for i, b in enumerate(batch) if not bool(b["cached"])], dtype=torch.long
        )
    return out
