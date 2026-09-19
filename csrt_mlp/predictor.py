"""Load a trained checkpoint and predict CSRT parameters for a frame + box."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .features import FrozenYoloxFeatures, torch_load
from .model import CSRTParamMLP
from .params_spec import denormalize, spec_from_json

__all__ = ["CSRTParamPredictor"]


class CSRTParamPredictor:
    def __init__(self, ckpt_path: str, yolox_ckpt: Optional[str] = None,
                 device: str = "cpu"):
        ckpt = torch_load(ckpt_path)
        fcfg = ckpt["feature_config"]
        mcfg = ckpt["mlp_config"]
        self.space = spec_from_json(ckpt["params_spec"])
        self.device = torch.device(device)

        # The MLP was trained on features from this exact backbone configuration.
        self.extractor = FrozenYoloxFeatures(
            ckpt=yolox_ckpt or ckpt["args"]["yolox_ckpt"],
            model_name=fcfg["yolox_name"],
            depth=fcfg.get("depth"),
            width=fcfg.get("width"),
            levels=fcfg["levels"],
            roi_size=fcfg["roi_size"],
            input_size=fcfg["input_size"],
            use_context=fcfg["use_context"],
            use_geometry=fcfg["use_geometry"],
            device=str(self.device),
        )
        if self.extractor.out_dim != fcfg["in_dim"]:
            raise RuntimeError(
                f"feature dim mismatch: checkpoint expects {fcfg['in_dim']}, "
                f"the extractor produces {self.extractor.out_dim}"
            )

        self.model = CSRTParamMLP(fcfg["in_dim"], mcfg["out_dim"],
                                  hidden=mcfg["hidden"], dropout=mcfg["dropout"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, box_xywh) -> Dict[str, float]:
        feats = self.extractor.extract_from_bgr(img_bgr, box_xywh)
        vec = self.model(feats.to(self.device)).cpu().numpy()[0]
        return denormalize(vec, self.space)

    @torch.no_grad()
    def predict_vector(self, img_bgr: np.ndarray, box_xywh) -> np.ndarray:
        feats = self.extractor.extract_from_bgr(img_bgr, box_xywh)
        return self.model(feats.to(self.device)).cpu().numpy()[0]
