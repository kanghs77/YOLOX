"""Frozen YOLOX-m feature extractor for CSRT-parameter regression.

The descriptor handed to the MLP is, per FPN level (strides 8 / 16 / 32):

  * ``roi_size x roi_size`` RoI-aligned features of the tracked box  -> target appearance
  * a globally average-pooled vector                                 -> scene context

plus 8 geometry features of the box (normalised position, size, aspect,
area ratio, sqrt-area).  Target size and scene texture are exactly what drives
``template_size`` / ``padding`` / ``scale_*``, so both halves matter.

The YOLOX weights are never updated: the module is put in ``eval()`` mode and
all parameters get ``requires_grad_(False)``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FrozenYoloxFeatures", "preprocess_image", "letterbox_scale", "torch_load"]

LEVEL_STRIDES = {"p3": 8, "p4": 16, "p5": 32}


# ---------------------------------------------------------------------------
# preprocessing (identical to YOLOX ValTransform, legacy=False: no /255)
# ---------------------------------------------------------------------------
def letterbox_scale(img_shape: Tuple[int, int], input_size: int) -> float:
    h, w = img_shape[:2]
    return min(input_size / float(h), input_size / float(w))


def preprocess_image(img_bgr: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
    """Resize keeping aspect ratio and pad bottom/right with 114.

    Returns the ``(3, S, S)`` float32 CHW array and the applied scale ``r`` -
    a box in original image coordinates maps to input coordinates by ``* r``
    (padding is bottom/right only, so there is no offset).
    """
    import cv2

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    r = min(input_size / float(h), input_size / float(w))
    nh, nw = int(round(h * r)), int(round(w * r))
    nh, nw = max(1, min(nh, input_size)), max(1, min(nw, input_size))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded[:nh, :nw] = resized
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return chw, float(r)


# ---------------------------------------------------------------------------
# RoI align (torchvision when present, grid_sample fallback otherwise)
# ---------------------------------------------------------------------------
def _roi_align_one_per_image(feat: torch.Tensor, boxes: torch.Tensor,
                             output_size: int, spatial_scale: float) -> torch.Tensor:
    """RoI-align with exactly one box per image.

    ``boxes`` is ``(B, 4)`` in *input-image* pixels (x1, y1, x2, y2).
    """
    B, C, H, W = feat.shape
    try:
        from torchvision.ops import roi_align as tv_roi_align

        idx = torch.arange(B, device=feat.device, dtype=feat.dtype).unsqueeze(1)
        rois = torch.cat([idx, boxes.to(feat.dtype)], dim=1)
        return tv_roi_align(feat, rois, (output_size, output_size),
                            spatial_scale=spatial_scale, sampling_ratio=2,
                            aligned=True)
    except ImportError:
        pass

    # --- fallback: bilinear sampling on a regular grid inside the box -----
    k = output_size
    b = boxes.to(feat.dtype) * spatial_scale                 # -> feature pixels
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x2 = torch.maximum(x2, x1 + 1e-3)
    y2 = torch.maximum(y2, y1 + 1e-3)

    steps = (torch.arange(k, device=feat.device, dtype=feat.dtype) + 0.5) / k
    xs = x1[:, None] + steps[None, :] * (x2 - x1)[:, None]   # (B, k)
    ys = y1[:, None] + steps[None, :] * (y2 - y1)[:, None]   # (B, k)
    # align_corners=False: normalised = (pix + 0.5) * 2 / size - 1
    gx = (xs + 0.5) * 2.0 / W - 1.0
    gy = (ys + 0.5) * 2.0 / H - 1.0
    grid = torch.stack(
        [gx[:, None, :].expand(B, k, k), gy[:, :, None].expand(B, k, k)], dim=-1
    )
    return F.grid_sample(feat, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------
def _build_backbone(model_name: str, depth: Optional[float], width: Optional[float]):
    """Prefer the installed `yolox` package, fall back to the vendored copy."""
    from .yolox_min import YOLOX_SIZES, YOLOPAFPN as MinYOLOPAFPN

    if depth is None or width is None:
        if model_name not in YOLOX_SIZES:
            raise ValueError(
                f"unknown model '{model_name}'; pass --depth/--width explicitly"
            )
        depth, width = YOLOX_SIZES[model_name]

    try:  # official package, if the user installed it
        from yolox.models import YOLOPAFPN  # type: ignore
    except Exception:
        YOLOPAFPN = MinYOLOPAFPN
    return YOLOPAFPN(depth, width), float(depth), float(width)


def torch_load(path: str):
    """``torch.load`` that works before and after the 2.6 ``weights_only`` flip.

    YOLOX checkpoints are full pickles (they carry numpy scalars and an optimiser
    state), so they cannot be read with ``weights_only=True``.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location="cpu")


def _load_yolox_state_dict(backbone: nn.Module, ckpt_path: str, strict: bool = False):
    ckpt = torch_load(ckpt_path)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("head."):
            continue                       # detection head is not used
        if k.startswith("backbone."):
            k = k[len("backbone."):]       # YOLOX.backbone == YOLOPAFPN
        cleaned[k] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=strict)
    loaded = len(backbone.state_dict()) - len(missing)
    if loaded == 0:
        raise RuntimeError(
            f"no weights matched when loading {ckpt_path} - is it a YOLOX checkpoint "
            f"of the requested size?"
        )
    return {"missing": list(missing), "unexpected": list(unexpected), "loaded": loaded}


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------
class FrozenYoloxFeatures(nn.Module):
    def __init__(
        self,
        ckpt: Optional[str] = None,
        model_name: str = "yolox-m",
        depth: Optional[float] = None,
        width: Optional[float] = None,
        levels: Sequence[str] = ("p3", "p4", "p5"),
        roi_size: int = 3,
        input_size: int = 640,
        use_context: bool = True,
        use_geometry: bool = True,
        device: str = "cpu",
        strict: bool = False,
    ):
        super().__init__()
        bad = [l for l in levels if l not in LEVEL_STRIDES]
        if bad:
            raise ValueError(f"unknown FPN level(s) {bad}; choose from {list(LEVEL_STRIDES)}")

        self.backbone, self.depth, self.width = _build_backbone(model_name, depth, width)
        self.load_report = None
        if ckpt:
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"YOLOX checkpoint not found: {ckpt}")
            self.load_report = _load_yolox_state_dict(self.backbone, ckpt, strict=strict)

        self.levels = list(levels)
        self.roi_size = int(roi_size)
        self.input_size = int(input_size)
        self.use_context = bool(use_context)
        self.use_geometry = bool(use_geometry)
        self.device = torch.device(device)

        # YOLOPAFPN returns (p3, p4, p5) with channels 256/512/1024 * width
        base = (256, 512, 1024)
        self._level_channels = {
            name: int(base[i] * self.width) for i, name in enumerate(("p3", "p4", "p5"))
        }

        self.freeze()
        self.to(self.device)

    # -- lifecycle ---------------------------------------------------------
    def freeze(self):
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        return self

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode whatever the trainer does."""
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def out_dim(self) -> int:
        dim = 0
        for name in self.levels:
            c = self._level_channels[name]
            dim += c * self.roi_size * self.roi_size
            if self.use_context:
                dim += c
        if self.use_geometry:
            dim += 8
        return dim

    # -- geometry ----------------------------------------------------------
    def _geometry(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """8 scale-aware descriptors of the box inside the padded input."""
        s = float(self.input_size)
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
        w = (x2 - x1).clamp(min=1e-3)
        h = (y2 - y1).clamp(min=1e-3)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        area = (w * h) / (s * s)
        return torch.stack(
            [
                cx / s, cy / s, w / s, h / s,
                torch.log(w / h),                  # log aspect ratio
                area,
                torch.sqrt(area),
                torch.log(torch.sqrt(w * h) + 1e-6) / 10.0,
            ],
            dim=1,
        )

    # -- forward -----------------------------------------------------------
    @torch.no_grad()
    def forward(self, images: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """``images``: (B, 3, S, S) preprocessed; ``boxes``: (B, 4) xyxy in input px."""
        images = images.to(self.device, non_blocking=True).float()
        boxes_xyxy = boxes_xyxy.to(self.device, non_blocking=True).float()

        feats = self.backbone(images)                    # (p3, p4, p5)
        by_name = dict(zip(("p3", "p4", "p5"), feats))

        parts: List[torch.Tensor] = []
        for name in self.levels:
            f = by_name[name]
            roi = _roi_align_one_per_image(
                f, boxes_xyxy, self.roi_size, 1.0 / LEVEL_STRIDES[name]
            )
            parts.append(roi.flatten(1))
            if self.use_context:
                parts.append(F.adaptive_avg_pool2d(f, 1).flatten(1))
        if self.use_geometry:
            parts.append(self._geometry(boxes_xyxy))
        return torch.cat(parts, dim=1)

    # -- convenience for single-image inference ----------------------------
    @torch.no_grad()
    def extract_from_bgr(self, img_bgr: np.ndarray, box_xywh) -> torch.Tensor:
        chw, r = preprocess_image(img_bgr, self.input_size)
        x, y, w, h = (float(v) for v in box_xywh)
        box = torch.tensor([[x * r, y * r, (x + w) * r, (y + h) * r]],
                           dtype=torch.float32)
        img = torch.from_numpy(chw).unsqueeze(0)
        return self.forward(img, box)
