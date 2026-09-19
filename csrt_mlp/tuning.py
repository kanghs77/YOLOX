"""Per-scene CSRT hyper-parameter tuning.

A *tuning unit* is a contiguous frame range of one sequence.  With
``granularity="video"`` there is a single unit per sequence (the whole clip);
with ``granularity="chunk"`` the clip is cut into fixed-length chunks so the
parameters - and hence the per-frame labels - vary inside a video.

Search = random search in the (log-)unit cube + Gaussian local refinement
around the incumbent.  Optuna's TPE sampler is used instead when available and
requested, which usually needs ~2x fewer trials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence as _Seq

import numpy as np

from .csrt_utils import track_sequence
from .params_spec import (
    CSRT_SEARCH_SPACE,
    ParamSpec,
    default_params,
    perturb_params,
    sample_params,
)

__all__ = ["TuneConfig", "TuningUnit", "make_units", "tune_unit", "tune_sequence"]


@dataclass
class TuneConfig:
    # --- search ----------------------------------------------------------
    n_trials: int = 40
    refine_trials: int = 20
    refine_sigma: float = 0.12
    sampler: str = "random"        # "random" | "tpe"
    seed: int = 0
    # --- unit definition --------------------------------------------------
    granularity: str = "video"     # "video" | "chunk"
    chunk_size: int = 120
    min_chunk: int = 30
    # --- evaluation -------------------------------------------------------
    frame_stride: int = 1          # sub-sample frames to speed tuning up
    max_frames: int = 300          # cap per unit (0 = no cap)
    reinit_iou: Optional[float] = 0.1
    reinit_skip: int = 5
    failure_penalty: float = 0.3
    min_gain: float = 0.0          # keep defaults if tuning gains less than this


@dataclass
class TuningUnit:
    seq_uid: str
    start: int
    end: int
    params: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    baseline_score: float = 0.0
    mean_iou: float = 0.0
    baseline_mean_iou: float = 0.0
    n_trials: int = 0
    seconds: float = 0.0
    used_default: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["gain"] = float(self.score - self.baseline_score)
        return d


def make_units(seq, cfg: TuneConfig) -> List[TuningUnit]:
    """Split one sequence into tuning units."""
    n = len(seq)
    if cfg.granularity == "video" or n <= cfg.chunk_size:
        return [TuningUnit(seq.uid, 0, n)]

    units: List[TuningUnit] = []
    for start in range(0, n, cfg.chunk_size):
        end = min(start + cfg.chunk_size, n)
        if end - start < cfg.min_chunk and units:
            units[-1].end = end          # glue a short tail onto the last chunk
        else:
            units.append(TuningUnit(seq.uid, start, end))
    return units


def _eval(seq, params, unit: TuningUnit, cfg: TuneConfig) -> dict:
    """Score one parameter set on one unit."""
    start, end = unit.start, unit.end
    stride = max(1, cfg.frame_stride)
    if cfg.max_frames and (end - start) // stride > cfg.max_frames:
        stride = max(stride, int(np.ceil((end - start) / cfg.max_frames)))
    res = track_sequence(
        seq,
        params=params,
        start=start,
        end=end,
        reinit_iou=cfg.reinit_iou,
        reinit_skip=cfg.reinit_skip,
        frame_stride=stride,
        failure_penalty=cfg.failure_penalty,
    )
    return {"score": res["score"], "mean_iou": res["mean_iou"]}


def _tpe_search(seq, unit, cfg, space, best):
    """Optuna TPE search; falls back to the caller's incumbent on failure."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {}
        for s in space:
            if s.kind in ("int", "odd_int"):
                params[s.name] = s.cast(
                    trial.suggest_int(s.name, int(s.low), int(s.high))
                )
            else:
                params[s.name] = trial.suggest_float(
                    s.name, s.low, s.high, log=s.log
                )
        return _eval(seq, params, unit, cfg)["score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed + unit.start),
    )
    study.enqueue_trial({s.name: best["params"][s.name] for s in space})
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    params = {s.name: s.cast(study.best_params[s.name]) for s in space}
    metrics = _eval(seq, params, unit, cfg)
    if metrics["score"] > best["score"]:
        best = {"params": params, **metrics}
    return best


def tune_unit(
    seq,
    unit: TuningUnit,
    cfg: TuneConfig,
    space: _Seq[ParamSpec] = None,
) -> TuningUnit:
    """Search the best CSRT parameters for one frame range."""
    space = list(space or CSRT_SEARCH_SPACE)
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed + abs(hash(unit.seq_uid)) % 100000 + unit.start)

    base_params = default_params(space)
    base = _eval(seq, base_params, unit, cfg)
    best = {"params": base_params, **base}
    n_done = 1

    if cfg.sampler == "tpe":
        try:
            best = _tpe_search(seq, unit, cfg, space, best)
            n_done += cfg.n_trials
        except ImportError:
            cfg = TuneConfig(**{**cfg.__dict__, "sampler": "random"})

    if cfg.sampler != "tpe":
        for _ in range(cfg.n_trials):
            cand = sample_params(rng, space)
            metrics = _eval(seq, cand, unit, cfg)
            n_done += 1
            if metrics["score"] > best["score"]:
                best = {"params": cand, **metrics}

    # local refinement around the incumbent (shrinking sigma)
    sigma = cfg.refine_sigma
    for i in range(cfg.refine_trials):
        cand = perturb_params(best["params"], rng, sigma=sigma, space=space)
        metrics = _eval(seq, cand, unit, cfg)
        n_done += 1
        if metrics["score"] > best["score"]:
            best = {"params": cand, **metrics}
        if (i + 1) % max(1, cfg.refine_trials // 3) == 0:
            sigma *= 0.5

    gain = best["score"] - base["score"]
    used_default = gain <= cfg.min_gain
    if used_default:
        best = {"params": base_params, **base}

    unit.params = {k: float(v) for k, v in best["params"].items()}
    unit.score = float(best["score"])
    unit.mean_iou = float(best["mean_iou"])
    unit.baseline_score = float(base["score"])
    unit.baseline_mean_iou = float(base["mean_iou"])
    unit.n_trials = n_done
    unit.seconds = float(time.time() - t0)
    unit.used_default = bool(used_default)
    return unit


def tune_sequence(seq, cfg: TuneConfig, space: _Seq[ParamSpec] = None) -> List[TuningUnit]:
    """Tune every unit of one sequence (entry point for the worker pool)."""
    units = make_units(seq, cfg)
    out = [tune_unit(seq, u, cfg, space) for u in units]
    seq.close()
    return out
