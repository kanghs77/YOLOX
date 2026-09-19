#!/usr/bin/env python3
"""3단계 - 고정 파라미터 CSRT vs MLP 예측 파라미터 CSRT 비교.

세 가지 전략을 같은 시퀀스/같은 프로토콜로 돌려 표로 비교한다.

    default : OpenCV 기본값 (기존 고정 변수 CSRT)
    mlp     : MLP 가 장면을 보고 예측한 파라미터
    oracle  : 1단계가 그 클립의 GT 를 보고 찾아낸 파라미터 (상한선)

사용 예::

    python tools/eval_tracker.py \
        --data-root datasets/OTB100 \
        --mlp-ckpt outputs/mlp/best.pth \
        --tuned-params outputs/csrt_labels/params_per_unit.json \
        --seqs-file outputs/mlp/val_sequences.txt \
        --output outputs/eval

``--output`` 을 주면 ``eval.json`` / ``eval.csv`` / ``eval.md`` 가 함께 저장된다.
반드시 2단계에서 **학습에 쓰이지 않은** 시퀀스로 평가해야 의미가 있다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csrt_mlp.csrt_utils import track_sequence  # noqa: E402
from csrt_mlp.datasets import discover_sequences  # noqa: E402
from csrt_mlp.metrics import aggregate, format_table, summarize_run, to_csv  # noqa: E402
from csrt_mlp.params_spec import default_params  # noqa: E402

STRATEGIES = ("default", "mlp", "oracle")


def parse_args():
    p = argparse.ArgumentParser(description="CSRT 파라미터 전략 비교")
    p.add_argument("--data-root", required=True)
    p.add_argument("--format", default="auto", choices=["auto", "otb", "mot", "video"])
    p.add_argument("--min-len", type=int, default=30)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--max-objects", type=int, default=0)
    p.add_argument("--seqs", nargs="*", default=None,
                   help="평가할 시퀀스 이름 (보통 2단계의 검증 split)")
    p.add_argument("--seqs-file", default=None,
                   help="시퀀스 이름이 한 줄에 하나씩 든 파일 (train_mlp.py 가 저장)")
    p.add_argument("--mlp-ckpt", default=None, help="학습된 MLP (best.pth)")
    p.add_argument("--yolox-ckpt", default=None,
                   help="MLP 체크포인트에 기록된 YOLOX 경로를 덮어쓸 때")
    p.add_argument("--tuned-params", default=None,
                   help="oracle 행을 위한 params_per_unit.json")
    p.add_argument("--predict-every", type=int, default=0,
                   help="N 프레임마다 파라미터를 다시 예측 (0 = 초기 프레임에서 한 번)")
    p.add_argument("--reinit-iou", type=float, default=-1.0,
                   help="이 IoU 미만이면 GT 로 재초기화. 음수면 one-pass(OPE)")
    p.add_argument("--reinit-skip", type=int, default=5)
    p.add_argument("--failure-penalty", type=float, default=0.3)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None,
                   help="결과를 저장할 디렉터리 또는 .json 경로")
    return p.parse_args()


def run_fixed(seq, params, args) -> dict:
    """고정 파라미터 한 벌로 시퀀스 전체를 추적."""
    reinit_iou = None if args.reinit_iou < 0 else args.reinit_iou
    return track_sequence(seq, params=params, reinit_iou=reinit_iou,
                          reinit_skip=args.reinit_skip,
                          frame_stride=args.frame_stride,
                          failure_penalty=args.failure_penalty)


def run_mlp(seq, predictor, args) -> dict:
    """MLP 예측 파라미터로 추적. ``--predict-every`` 면 구간마다 다시 예측."""
    reinit_iou = None if args.reinit_iou < 0 else args.reinit_iou

    if args.predict_every <= 0:
        img = seq.read(0)
        params = predictor.predict(img, seq.gt[0])
        res = run_fixed(seq, params, args)
        res["params_used"] = [params]
        return res

    ious: List[float] = []
    errors: List[float] = []
    failures, used, fps_w = 0, [], []
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
            ious.extend(res["ious"])
            errors.extend(res["center_errors"])
            failures += res["n_failures"]
            fps_w.append((res["fps"], res["n_eval"]))

    n = len(ious)
    mean_iou = float(np.mean(ious)) if n else 0.0
    fps = (sum(f * w for f, w in fps_w) / sum(w for _, w in fps_w)) if fps_w else 0.0
    return {
        "mean_iou": mean_iou, "ious": ious, "center_errors": errors,
        "n_failures": failures, "n_eval": n, "fps": fps,
        "score": mean_iou - args.failure_penalty * (failures / max(n, 1)),
        "params_used": used,
    }


def main():
    args = parse_args()

    sequences = discover_sequences(args.data_root, fmt=args.format,
                                   min_len=args.min_len,
                                   max_objects=args.max_objects or None)
    wanted = set(args.seqs or [])
    if args.seqs_file:
        wanted |= {ln.strip() for ln in Path(args.seqs_file).read_text().splitlines()
                   if ln.strip()}
    found_all = {s.name for s in sequences} | {s.uid for s in sequences}
    if wanted:
        sequences = [s for s in sequences if s.name in wanted or s.uid in wanted]
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    if not sequences:
        msg = [f"{args.data_root} 에서 평가할 시퀀스를 찾지 못했습니다."]
        if wanted:
            msg.append(f"  요청한 시퀀스 {len(wanted)}개: {sorted(wanted)[:10]}")
            msg.append(f"  이 경로에 있는 시퀀스 {len(found_all)}개: {sorted(found_all)[:10]}")
            msg.append("  --data-root 가 검증 시퀀스를 포함하는 상위 경로인지 확인하세요 "
                       "(1단계와 같은 경로여야 합니다).")
        else:
            msg.append("  python tools/check_dataset.py --data-root <경로> 로 배치를 확인하세요.")
        raise SystemExit("\n".join(msg))

    if wanted:
        missing = sorted(wanted - found_all)
        if missing:
            print(f"경고: 요청한 시퀀스 중 {len(missing)}개를 찾지 못했습니다: {missing[:10]}")

    tuned: Dict[str, List[dict]] = {}
    if args.tuned_params:
        tuned = json.loads(Path(args.tuned_params).read_text())

    predictor = None
    if args.mlp_ckpt:
        from csrt_mlp.predictor import CSRTParamPredictor

        print(f"MLP 로드: {args.mlp_ckpt}")
        predictor = CSRTParamPredictor(args.mlp_ckpt, args.yolox_ckpt, args.device)

    print(f"평가 시퀀스 {len(sequences)}개 | "
          f"프로토콜 {'OPE(재초기화 없음)' if args.reinit_iou < 0 else f'reinit@{args.reinit_iou}'}\n")

    rows: List[dict] = []
    t_start = time.time()
    for i, seq in enumerate(sequences, 1):
        row = {"seq": seq.uid, "n_frames": len(seq)}

        row["default"] = summarize_run(run_fixed(seq, default_params(), args))

        if predictor is not None:
            res = run_mlp(seq, predictor, args)
            row["mlp"] = summarize_run(res)
            row["mlp_params"] = res["params_used"][0] if res["params_used"] else {}

        if seq.uid in tuned and tuned[seq.uid]:
            unit = max(tuned[seq.uid], key=lambda u: u["end"] - u["start"])
            row["oracle"] = summarize_run(run_fixed(seq, unit["params"], args))

        seq.close()
        rows.append(row)
        parts = [f"{k} {row[k]['mean_iou']:.3f}" for k in STRATEGIES if k in row]
        print(f"[{i}/{len(sequences)}] {seq.uid:<34} " + " | ".join(parts), flush=True)

    summary = aggregate(rows, STRATEGIES)
    table = format_table(rows, summary, STRATEGIES, baseline="default")
    print("\n" + table)
    print(f"\n총 소요 {time.time() - t_start:.1f}s")

    if args.output:
        out = Path(args.output)
        if out.suffix == ".json":
            json_path, base = out, out.with_suffix("")
        else:
            out.mkdir(parents=True, exist_ok=True)
            json_path, base = out / "eval.json", out / "eval"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(
            {"summary": summary, "per_sequence": rows, "config": vars(args)}, indent=2))
        Path(f"{base}.csv").write_text(to_csv(rows, [s for s in STRATEGIES if s in summary]))
        Path(f"{base}.md").write_text("```\n" + table + "\n```\n")
        print(f"저장: {json_path}, {base}.csv, {base}.md")


if __name__ == "__main__":
    main()
