#!/usr/bin/env python3
"""Stage 3 - compare CSRT with default / oracle-tuned / MLP-predicted parameters.

Example
-------
    python tools/eval_tracker.py \
        --data-root /data/tracking \
        --mlp-ckpt outputs/mlp/best.pth \
        --yolox-ckpt weights/yolox_m.pth \
        --tuned-params outputs/csrt_labels/params_per_unit.json \
        --output outputs/eval.json --device cuda

``oracle`` is the stage-1 result on that same clip - an upper bound the MLP is
trying to reach without having seen the clip's ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csrt_mlp.csrt_utils import track_sequence  # noqa: E402
from csrt_mlp.datasets import discover_sequences  # noqa: E402
from csrt_mlp.params_spec import default_params  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CSRT parameter strategies")
    p.add_argument("--data-root", required=True)
    p.add_argument("--format", default="auto", choices=["auto", "otb", "mot", "video"])
    p.add_argument("--min-len", type=int, default=30)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-objects", type=int, default=0)
    p.add_argument("--seqs", nargs="*", default=None,
                   help="evaluate only these sequence names (e.g. the val split)")
    p.add_argument("--mlp-ckpt", default=None, help="trained MLP (best.pth)")
    p.add_argument("--yolox-ckpt", default=None,
                   help="override the YOLOX checkpoint path stored in the MLP ckpt")
    p.add_argument("--tuned-params", default=None,
                   help="params_per_unit.json for the oracle row")
    p.add_argument("--predict-every", type=int, default=0,
                   help="re-predict parameters every N frames and restart the tracker "
                        "(0 = predict once on the init frame)")
    p.add_argument("--reinit-iou", type=float, default=-1.0,
                   help="re-init below this IoU; <0 = one-pass evaluation (OPE)")
    p.add_argument("--reinit-skip", type=int, default=5)
    p.add_argument("--failure-penalty", type=float, default=0.3)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    return p.parse_args()


def track_with_predictor(seq, predictor, args) -> dict:
    """Run CSRT with MLP-predicted parameters, optionally refreshed every N frames."""
    reinit_iou = None if args.reinit_iou < 0 else args.reinit_iou
    if args.predict_every <= 0:
        img = seq.read(0)
        params = predictor.predict(img, seq.gt[0])
        res = track_sequence(seq, params=params, reinit_iou=reinit_iou,
                             reinit_skip=args.reinit_skip,
                             frame_stride=args.frame_stride,
                             failure_penalty=args.failure_penalty)
        res["params_used"] = [params]
        return res

    # segment-wise: re-read the scene every N frames and restart with fresh params
    ious, failures, used = [], 0, []
    for start in range(0, len(seq), args.predict_every):
        end = min(start + args.predict_every, len(seq))
        if end - start < 2:
            continue
        box = seq.gt[start]
        if not np.all(np.isfinite(box)) or box[2] <= 1 or box[3] <= 1:
            continue
        img = seq.read(start)
        if img is None:
            continue
        params = predictor.predict(img, box)
        used.append(params)
        res = track_sequence(seq, params=params, start=start, end=end,
                             reinit_iou=reinit_iou, reinit_skip=args.reinit_skip,
                             frame_stride=args.frame_stride,
                             failure_penalty=args.failure_penalty)
        if res["n_eval"]:
            ious.extend([res["mean_iou"]] * res["n_eval"])
            failures += res["n_failures"]
    n = len(ious)
    mean_iou = float(np.mean(ious)) if n else 0.0
    return {"mean_iou": mean_iou, "success_rate": float("nan"), "n_failures": failures,
            "n_eval": n, "score": mean_iou - args.failure_penalty * (failures / max(n, 1)),
            "params_used": used}


def main():
    args = parse_args()
    reinit_iou = None if args.reinit_iou < 0 else args.reinit_iou

    sequences = discover_sequences(args.data_root, fmt=args.format,
                                   min_len=args.min_len,
                                   max_objects=args.max_objects or None)
    if args.seqs:
        wanted = set(args.seqs)
        sequences = [s for s in sequences if s.name in wanted]
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(f"no sequences found under {args.data_root}")

    tuned: Dict[str, List[dict]] = {}
    if args.tuned_params:
        tuned = json.loads(Path(args.tuned_params).read_text())

    predictor = None
    if args.mlp_ckpt:
        from csrt_mlp.predictor import CSRTParamPredictor

        print(f"loading MLP from {args.mlp_ckpt} ...")
        predictor = CSRTParamPredictor(args.mlp_ckpt, args.yolox_ckpt, args.device)

    rows: List[dict] = []
    for i, seq in enumerate(sequences, 1):
        row = {"seq": seq.uid, "n_frames": len(seq)}

        res = track_sequence(seq, params=default_params(), reinit_iou=reinit_iou,
                             reinit_skip=args.reinit_skip,
                             frame_stride=args.frame_stride,
                             failure_penalty=args.failure_penalty)
        row["default"] = {k: res[k] for k in
                          ("mean_iou", "success_rate", "n_failures", "score")}

        if seq.uid in tuned and tuned[seq.uid]:
            # the whole-clip unit if there is one, otherwise the first chunk
            unit = max(tuned[seq.uid], key=lambda u: u["end"] - u["start"])
            res = track_sequence(seq, params=unit["params"], reinit_iou=reinit_iou,
                                 reinit_skip=args.reinit_skip,
                                 frame_stride=args.frame_stride,
                                 failure_penalty=args.failure_penalty)
            row["oracle"] = {k: res[k] for k in
                             ("mean_iou", "success_rate", "n_failures", "score")}

        if predictor is not None:
            res = track_with_predictor(seq, predictor, args)
            row["mlp"] = {k: res[k] for k in
                          ("mean_iou", "success_rate", "n_failures", "score")}
            row["mlp_params"] = res["params_used"][0] if res["params_used"] else {}

        seq.close()
        rows.append(row)
        parts = [f"{k} IoU {row[k]['mean_iou']:.3f}"
                 for k in ("default", "oracle", "mlp") if k in row]
        print(f"[{i}/{len(sequences)}] {seq.uid}: " + " | ".join(parts), flush=True)

    summary = {}
    for key in ("default", "oracle", "mlp"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            summary[key] = {
                "mean_iou": float(np.mean([v["mean_iou"] for v in vals])),
                "success_rate": float(np.nanmean([v["success_rate"] for v in vals])),
                "score": float(np.mean([v["score"] for v in vals])),
                "n_sequences": len(vals),
            }

    print("\n=== summary =====================================================")
    print(f"{'strategy':10s} {'mean IoU':>10s} {'success':>10s} {'score':>10s} {'#seq':>6s}")
    for key, v in summary.items():
        print(f"{key:10s} {v['mean_iou']:10.4f} {v['success_rate']:10.4f} "
              f"{v['score']:10.4f} {v['n_sequences']:6d}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump({"summary": summary, "per_sequence": rows,
                       "config": vars(args)}, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
