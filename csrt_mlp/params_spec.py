"""CSRT hyper-parameter search space and (de)normalisation utilities.

The same spec object is used in three places:

1. ``tools/tune_csrt.py``  - to sample candidate parameters per scene.
2. ``tools/train_mlp.py``  - to turn the tuned parameters into a normalised
   regression target in ``[0, 1]`` (and back).
3. ``tools/eval_tracker.py`` - to decode the MLP prediction into a real
   ``cv2.TrackerCSRT_Params`` object.

Only parameters that are numeric *and* have a real influence on CSRT accuracy
are tuned.  Boolean switches (``use_hog``, ``use_color_names`` ...) are kept at
their OpenCV defaults so the regression target stays continuous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence as _Seq

import numpy as np

__all__ = [
    "ParamSpec",
    "CSRT_SEARCH_SPACE",
    "param_names",
    "default_params",
    "sample_params",
    "perturb_params",
    "normalize",
    "denormalize",
    "spec_to_json",
    "spec_from_json",
]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable CSRT parameter."""

    name: str
    low: float
    high: float
    default: float
    kind: str = "float"  # "float" | "int" | "odd_int"
    log: bool = False    # sample / normalise in log space

    # -- value <-> unit interval ------------------------------------------
    def to_unit(self, value: float) -> float:
        lo, hi, v = self.low, self.high, float(value)
        v = min(max(v, lo), hi)
        if self.log:
            return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return (v - lo) / (hi - lo)

    def from_unit(self, u: float) -> float:
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return self.cast(v)

    def cast(self, v: float) -> float:
        v = min(max(float(v), self.low), self.high)
        if self.kind == "int":
            return int(round(v))
        if self.kind == "odd_int":
            iv = int(round(v))
            if iv % 2 == 0:  # CSRT expects an odd number of scales
                iv += 1 if iv + 1 <= self.high else -1
            return int(iv)
        return float(v)


# ---------------------------------------------------------------------------
# The search space.  Defaults mirror cv2.TrackerCSRT_Params() (OpenCV 4.x).
# ---------------------------------------------------------------------------
CSRT_SEARCH_SPACE: List[ParamSpec] = [
    # --- spatial / template geometry -------------------------------------
    ParamSpec("padding",              1.0,   4.0,   3.0),
    ParamSpec("template_size",      100.0, 320.0, 200.0),
    ParamSpec("gsl_sigma",            0.4,   2.5,   1.0),
    # --- correlation filter ----------------------------------------------
    ParamSpec("filter_lr",           0.005, 0.15,  0.02, log=True),
    ParamSpec("weights_lr",          0.005, 0.10,  0.02, log=True),
    ParamSpec("admm_iterations",      2,     6,     4,   kind="int"),
    ParamSpec("psr_threshold",       0.015, 0.15,  0.035, log=True),
    # --- HoG ---------------------------------------------------------------
    ParamSpec("num_hog_channels_used", 9,   18,    18,   kind="int"),
    ParamSpec("hog_clip",            0.10,  0.50,  0.20),
    # --- colour histogram / segmentation ----------------------------------
    ParamSpec("histogram_lr",        0.005, 0.20,  0.04, log=True),
    ParamSpec("background_ratio",     1,     4,     2,   kind="int"),
    # --- scale estimation --------------------------------------------------
    ParamSpec("number_of_scales",    11,    53,    33,   kind="odd_int"),
    ParamSpec("scale_lr",            0.005, 0.20,  0.025, log=True),
    ParamSpec("scale_step",          1.005, 1.08,  1.02),
    ParamSpec("scale_sigma_factor",  0.05,  0.50,  0.25),
]

_SPEC_BY_NAME = {s.name: s for s in CSRT_SEARCH_SPACE}


def param_names(space: _Seq[ParamSpec] = None) -> List[str]:
    return [s.name for s in (space or CSRT_SEARCH_SPACE)]


def default_params(space: _Seq[ParamSpec] = None) -> Dict[str, float]:
    return {s.name: s.cast(s.default) for s in (space or CSRT_SEARCH_SPACE)}


def sample_params(rng: np.random.Generator, space: _Seq[ParamSpec] = None) -> Dict[str, float]:
    """Uniform sample in the (possibly log) unit cube."""
    space = space or CSRT_SEARCH_SPACE
    return {s.name: s.from_unit(float(rng.random())) for s in space}


def perturb_params(
    params: Dict[str, float],
    rng: np.random.Generator,
    sigma: float = 0.12,
    space: _Seq[ParamSpec] = None,
) -> Dict[str, float]:
    """Gaussian jitter in unit space - used by the local refinement stage."""
    space = space or CSRT_SEARCH_SPACE
    out = {}
    for s in space:
        u = s.to_unit(params.get(s.name, s.default))
        out[s.name] = s.from_unit(u + float(rng.normal(0.0, sigma)))
    return out


def normalize(params: Dict[str, float], space: _Seq[ParamSpec] = None) -> np.ndarray:
    space = space or CSRT_SEARCH_SPACE
    return np.array(
        [s.to_unit(params.get(s.name, s.default)) for s in space], dtype=np.float32
    )


def denormalize(vec, space: _Seq[ParamSpec] = None) -> Dict[str, float]:
    space = space or CSRT_SEARCH_SPACE
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if vec.shape[0] != len(space):
        raise ValueError(f"expected {len(space)} values, got {vec.shape[0]}")
    return {s.name: s.from_unit(float(v)) for s, v in zip(space, vec)}


# ---------------------------------------------------------------------------
# Serialisation - stage 1 stores the space it used so stage 2/3 decode the
# MLP output with exactly the same ranges.
# ---------------------------------------------------------------------------
def spec_to_json(space: _Seq[ParamSpec] = None) -> List[dict]:
    return [asdict(s) for s in (space or CSRT_SEARCH_SPACE)]


def spec_from_json(data) -> List[ParamSpec]:
    return [ParamSpec(**d) for d in data]
