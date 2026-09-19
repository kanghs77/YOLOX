"""Version-tolerant CSRT construction + tracking evaluation on a GT sequence."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # cv2 is only needed at run time, not at import time of the package
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

__all__ = [
    "make_csrt_params",
    "create_csrt",
    "iou_xywh",
    "track_sequence",
    "TrackResult",
]

_WARNED_UNKNOWN = set()


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not available. Install `opencv-contrib-python` "
            "(the plain `opencv-python` wheel does not ship TrackerCSRT)."
        )


def make_csrt_params(overrides: Optional[Dict[str, float]] = None):
    """Build a ``TrackerCSRT_Params`` and apply ``overrides``.

    Works with both the modern (``cv2.TrackerCSRT_Params``) and the legacy
    (``cv2.legacy.TrackerCSRT_Params``) contrib layouts.
    """
    _require_cv2()
    if hasattr(cv2, "TrackerCSRT_Params"):
        params = cv2.TrackerCSRT_Params()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_Params"):
        params = cv2.legacy.TrackerCSRT_Params()
    else:  # pragma: no cover
        raise RuntimeError(
            "TrackerCSRT_Params not found - install `opencv-contrib-python`."
        )

    for key, value in (overrides or {}).items():
        if not hasattr(params, key):
            if key not in _WARNED_UNKNOWN:
                _WARNED_UNKNOWN.add(key)
                warnings.warn(f"CSRT parameter '{key}' unknown in this OpenCV build - ignored")
            continue
        current = getattr(params, key)
        if isinstance(current, bool):
            setattr(params, key, bool(value))
        elif isinstance(current, int):
            setattr(params, key, int(round(float(value))))
        elif isinstance(current, float):
            setattr(params, key, float(value))
        else:
            setattr(params, key, value)
    return params


def create_csrt(params: Optional[Dict[str, float]] = None):
    """Instantiate a CSRT tracker for any OpenCV >= 4.0 build."""
    _require_cv2()
    p = make_csrt_params(params)
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create(p)
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create(p)
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create(p)
    raise RuntimeError("No TrackerCSRT factory found in this OpenCV build.")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def iou_xywh(a, b) -> float:
    """IoU of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _valid(box) -> bool:
    box = np.asarray(box, dtype=np.float64)
    return bool(np.all(np.isfinite(box)) and box[2] > 1 and box[3] > 1)


def _as_rect(box, img_shape) -> Tuple[int, int, int, int]:
    """Clamp a float box to a valid integer rect inside the image."""
    h, w = img_shape[:2]
    x, y, bw, bh = (float(v) for v in box)
    x = min(max(x, 0.0), w - 2.0)
    y = min(max(y, 0.0), h - 2.0)
    bw = min(max(bw, 2.0), w - x)
    bh = min(max(bh, 2.0), h - y)
    return int(round(x)), int(round(y)), int(round(bw)), int(round(bh))


class TrackResult(dict):
    """Plain dict with attribute access for readability."""

    __getattr__ = dict.__getitem__


def track_sequence(
    seq,
    params: Optional[Dict[str, float]] = None,
    start: int = 0,
    end: Optional[int] = None,
    reinit_iou: Optional[float] = None,
    reinit_skip: int = 5,
    frame_stride: int = 1,
    collect_boxes: bool = False,
    failure_penalty: float = 0.3,
) -> TrackResult:
    """Run CSRT over ``seq[start:end]`` and score it against the ground truth.

    Parameters
    ----------
    seq
        A :class:`csrt_mlp.datasets.Sequence`.
    reinit_iou
        If not ``None``, the tracker is re-initialised from the ground truth
        whenever IoU drops below this value (VOT-style reset).  ``None`` gives a
        plain one-pass evaluation (OPE).
    failure_penalty
        Weight of the reset rate in the scalar ``score`` used by the tuner.

    Returns
    -------
    TrackResult with ``mean_iou``, ``success_rate`` (IoU > 0.5), ``n_failures``,
    ``n_eval``, ``score`` and the raw per-frame ``ious``.
    """
    end = len(seq) if end is None else min(end, len(seq))
    if end - start < 2:
        return TrackResult(mean_iou=0.0, success_rate=0.0, n_failures=0,
                           n_eval=0, score=0.0, ious=[], boxes=[])

    tracker = None
    ious: List[float] = []
    boxes: List[Optional[Tuple[float, float, float, float]]] = []
    n_failures = 0
    skip_until = -1

    for idx in range(start, end, max(1, frame_stride)):
        gt = seq.gt[idx]
        img = None

        if tracker is None:
            # (re-)initialise on the first usable ground-truth box
            if not _valid(gt):
                boxes.append(None)
                continue
            img = seq.read(idx)
            if img is None:
                boxes.append(None)
                continue
            tracker = create_csrt(params)
            tracker.init(img, _as_rect(gt, img.shape))
            boxes.append(tuple(float(v) for v in gt))
            continue

        if idx < skip_until:
            boxes.append(None)
            continue

        img = seq.read(idx)
        if img is None:
            boxes.append(None)
            continue

        ok, box = tracker.update(img)
        box = tuple(float(v) for v in box) if box is not None else None
        boxes.append(box if ok else None)

        if not _valid(gt):
            continue  # target absent / occluded -> not scored

        iou = iou_xywh(box, gt) if (ok and box is not None) else 0.0
        ious.append(iou)

        if reinit_iou is not None and iou < reinit_iou:
            n_failures += 1
            tracker = None            # forces re-init at the next valid GT
            skip_until = idx + reinit_skip

    n_eval = len(ious)
    if n_eval == 0:
        return TrackResult(mean_iou=0.0, success_rate=0.0, n_failures=n_failures,
                           n_eval=0, score=0.0, ious=[],
                           boxes=boxes if collect_boxes else [])

    mean_iou = float(np.mean(ious))
    success = float(np.mean([i > 0.5 for i in ious]))
    score = mean_iou - failure_penalty * (n_failures / n_eval)
    return TrackResult(
        mean_iou=mean_iou,
        success_rate=success,
        n_failures=n_failures,
        n_eval=n_eval,
        score=float(score),
        ious=ious,
        boxes=boxes if collect_boxes else [],
    )
