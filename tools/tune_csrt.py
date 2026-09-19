#!/usr/bin/env python3
"""Stage 1 - tune CSRT per scene and dump the best parameters for every frame.

Example
-------
    python tools/tune_csrt.py \
        --data-root /data/tracking \
        --output outputs/csrt_labels \
        --granularity chunk --chunk-size 120 \
        --n-trials 48 --refine-trials 24 --workers 8

Outputs (under ``--output``)
----------------------------
``labels.jsonl``      one JSON record per annotated frame (image, box, params)
``params_per_unit.json``  the tuned parameters of every scene / chunk
``params_spec.json``  the search space used (stage 2/3 decode with it)
``summary.json``      default vs tuned accuracy, per sequence and overall
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csrt_mlp.datasets import discover_sequences, dump_video_frames  # noqa: E402
from csrt_mlp.params_spec import (  # noqa: E402
    CSRT_SEARCH_SPACE,
    normalize,
    param_names,
    spec_to_json,
)
from csrt_mlp.tuning import TuneConfig, tune_sequence  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Per-scene CSRT hyper-parameter tuning")
    # data
    p.add_argument("--data-root", required=True,
                   help="dataset root (OTB/LaSOT/GOT-10k/MOT folder, or a video file)")
    p.add_argument("--format", default="auto", choices=["auto", "otb", "mot", "video"])
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--min-len", type=int, default=30, help="skip shorter sequences")
    p.add_argument("--max-sequences", type=int, default=0, help="0 = all")
    p.add_argument("--max-objects", type=int, default=0,
                   help="MOT only: keep at most N identities per sequence (0 = all)")
    p.add_argument("--min-visibility", type=float, default=0.3, help="MOT only")
    p.add_argument("--dump-frames", action="store_true",
                   help="decode video sequences to JPEG so stage 2 reads images directly")
    # search
    p.add_argument("--granularity", default="video", choices=["video", "chunk"],
                   help="one parameter set per clip, or one per fixed-length chunk")
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--min-chunk", type=int, default=30)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--refine-trials", type=int, default=20)
    p.add_argument("--refine-sigma", type=float, default=0.12)
    p.add_argument("--sampler", default="random", choices=["random", "tpe"],
                   help="tpe needs `optuna`")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-gain", type=float, default=0.0,
                   help="keep the OpenCV defaults when tuning gains less than this")
    # evaluation during search
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=300,
                   help="cap frames evaluated per trial (0 = no cap)")
    p.add_argument("--reinit-iou", type=float, default=0.1,
                   help="re-init from GT below this IoU; <0 disables (pure OPE)")
    p.add_argument("--reinit-skip", type=int, default=5)
    p.add_argument("--failure-penalty", type=float, default=0.3)
    # runtime
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--sidecar", action="store_true",
                   help="also write <image>.csrt.json next to every frame")
    return p.parse_args()


def _worker(payload):
    seq, cfg = payload
    try:
        return seq.uid, [u.to_dict() for u in tune_sequence(seq, cfg)], None
    except Exception as exc:  # keep one bad clip from killing the whole run
        return seq.uid, [], f"{type(exc).__name__}: {exc}"


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TuneConfig(
        n_trials=args.n_trials,
        refine_trials=args.refine_trials,
        refine_sigma=args.refine_sigma,
        sampler=args.sampler,
        seed=args.seed,
        granularity=args.granularity,
        chunk_size=args.chunk_size,
        min_chunk=args.min_chunk,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        reinit_iou=None if args.reinit_iou < 0 else args.reinit_iou,
        reinit_skip=args.reinit_skip,
        failure_penalty=args.failure_penalty,
        min_gain=args.min_gain,
    )

    print(f"[1/4] scanning {args.data_root} ...", flush=True)
    sequences = discover_sequences(
        args.data_root,
        fmt=args.format,
        min_len=args.min_len,
        max_objects=args.max_objects or None,
        min_visibility=args.min_visibility,
    )
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise SystemExit(
            f"no sequences found under {args.data_root}. Expected an image folder + "
            f"groundtruth txt, a MOT-style img1/+gt/gt.txt, or a video with a sibling "
            f"groundtruth txt."
        )
    print(f"      {len(sequences)} sequence(s), "
          f"{sum(len(s) for s in sequences)} annotated frames")

    if args.dump_frames:
        frames_dir = out_dir / "frames"
        sequences = [
            dump_video_frames(s, str(frames_dir)) if s.video_path else s
            for s in sequences
        ]
        print(f"      decoded video frames into {frames_dir}")

    seq_by_uid = {s.uid: s for s in sequences}

    print(f"[2/4] tuning ({args.sampler}, {args.n_trials}+{args.refine_trials} trials/unit, "
          f"{args.workers} worker(s)) ...", flush=True)
    t0 = time.time()
    results: Dict[str, List[dict]] = {}
    failures: Dict[str, str] = {}

    payloads = [(s, cfg) for s in sequences]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for i, fut in enumerate(as_completed(futures), 1):
                uid, units, err = fut.result()
                if err:
                    failures[uid] = err
                    print(f"      [{i}/{len(futures)}] {uid}: FAILED ({err})", flush=True)
                    continue
                results[uid] = units
                best = np.mean([u["score"] for u in units]) if units else 0.0
                base = np.mean([u["baseline_score"] for u in units]) if units else 0.0
                print(f"      [{i}/{len(futures)}] {uid}: score {base:.3f} -> {best:.3f} "
                      f"({len(units)} unit(s))", flush=True)
    else:
        for i, p in enumerate(payloads, 1):
            uid, units, err = _worker(p)
            if err:
                failures[uid] = err
                print(f"      [{i}/{len(payloads)}] {uid}: FAILED ({err})", flush=True)
                continue
            results[uid] = units
            best = np.mean([u["score"] for u in units]) if units else 0.0
            base = np.mean([u["baseline_score"] for u in units]) if units else 0.0
            print(f"      [{i}/{len(payloads)}] {uid}: score {base:.3f} -> {best:.3f} "
                  f"({len(units)} unit(s))", flush=True)

    tune_seconds = time.time() - t0
    if not results:
        raise SystemExit("every sequence failed to tune - see the messages above")

    # ------------------------------------------------------------------
    # per-frame labels
    # ------------------------------------------------------------------
    print("[3/4] writing per-frame labels ...", flush=True)
    names = param_names()
    labels_path = out_dir / "labels.jsonl"
    n_records = 0
    sidecars: Dict[str, dict] = {}

    with open(labels_path, "w") as fh:
        for uid, units in sorted(results.items()):
            seq = seq_by_uid[uid]
            for unit in units:
                vec = normalize(unit["params"]).tolist()
                for idx in range(unit["start"], unit["end"]):
                    box = seq.gt[idx]
                    if not np.all(np.isfinite(box)) or box[2] <= 1 or box[3] <= 1:
                        continue  # no target in this frame -> nothing to condition on
                    ref = seq.frame_ref(idx)
                    rec = {
                        "seq": seq.name,
                        "uid": uid,
                        "track_id": seq.track_id,
                        "unit": f"{uid}@{unit['start']}-{unit['end']}",
                        "frame_index": idx,
                        "image": ref["image"],
                        "video": ref["video"],
                        "frame_no": ref["frame_no"],
                        "bbox": [float(v) for v in box],
                        "params": unit["params"],
                        "param_names": names,
                        "param_vec": vec,
                        "score": unit["score"],
                        "baseline_score": unit["baseline_score"],
                        "gain": unit["score"] - unit["baseline_score"],
                        "used_default": unit["used_default"],
                    }
                    fh.write(json.dumps(rec) + "\n")
                    n_records += 1
                    if args.sidecar and ref["image"]:
                        sidecars[ref["image"]] = rec
            seq.close()

    if args.sidecar:
        for image_path, rec in sidecars.items():
            try:
                with open(f"{image_path}.csrt.json", "w") as fh:
                    json.dump(rec, fh)
            except OSError as exc:
                print(f"      warning: sidecar for {image_path} failed ({exc})")
        print(f"      wrote {len(sidecars)} sidecar files")

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    print("[4/4] writing summary ...", flush=True)
    with open(out_dir / "params_per_unit.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(out_dir / "params_spec.json", "w") as fh:
        json.dump(spec_to_json(CSRT_SEARCH_SPACE), fh, indent=2)

    all_units = [u for units in results.values() for u in units]
    per_seq = {
        uid: {
            "n_units": len(units),
            "baseline_mean_iou": float(np.mean([u["baseline_mean_iou"] for u in units])),
            "tuned_mean_iou": float(np.mean([u["mean_iou"] for u in units])),
            "baseline_score": float(np.mean([u["baseline_score"] for u in units])),
            "tuned_score": float(np.mean([u["score"] for u in units])),
        }
        for uid, units in results.items() if units
    }
    summary = {
        "config": vars(args),
        "n_sequences": len(results),
        "n_units": len(all_units),
        "n_frame_labels": n_records,
        "tuning_seconds": tune_seconds,
        "baseline_mean_iou": float(np.mean([u["baseline_mean_iou"] for u in all_units])),
        "tuned_mean_iou": float(np.mean([u["mean_iou"] for u in all_units])),
        "baseline_score": float(np.mean([u["baseline_score"] for u in all_units])),
        "tuned_score": float(np.mean([u["score"] for u in all_units])),
        "units_kept_default": int(sum(u["used_default"] for u in all_units)),
        "per_sequence": per_seq,
        "failures": failures,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"\ndone in {tune_seconds:.1f}s\n"
        f"  units          : {summary['n_units']} "
        f"({summary['units_kept_default']} kept the OpenCV defaults)\n"
        f"  frame labels   : {n_records} -> {labels_path}\n"
        f"  mean IoU       : {summary['baseline_mean_iou']:.4f} (default) -> "
        f"{summary['tuned_mean_iou']:.4f} (tuned)\n"
        f"  tuner score    : {summary['baseline_score']:.4f} -> {summary['tuned_score']:.4f}"
    )
    if failures:
        print(f"  failed clips   : {len(failures)} (see summary.json)")


if __name__ == "__main__":
    main()
