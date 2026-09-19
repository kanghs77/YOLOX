#!/usr/bin/env python3
"""데이터셋을 올바른 위치에 두었는지 검사한다.

    python tools/check_dataset.py --data-root datasets/OTB100

각 시퀀스를 실제로 열어 첫 프레임을 읽고 GT 박스가 이미지 범위 안에 있는지까지
확인하므로, 학습을 돌리기 전에 배치가 맞는지 바로 알 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csrt_mlp.datasets import discover_sequences  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="데이터셋 배치 검사")
    p.add_argument("--data-root", required=True, help="데이터셋 루트 경로")
    p.add_argument("--format", default="auto", choices=["auto", "otb", "mot", "video"])
    p.add_argument("--min-len", type=int, default=30, help="이보다 짧은 시퀀스는 제외")
    p.add_argument("--max-objects", type=int, default=0, help="MOT 전용: ID 수 제한 (0=전체)")
    p.add_argument("--min-visibility", type=float, default=0.3, help="MOT 전용")
    p.add_argument("--list", action="store_true", help="모든 시퀀스를 나열")
    p.add_argument("--check-frames", type=int, default=3,
                   help="시퀀스마다 실제로 읽어볼 프레임 수 (0=읽지 않음)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f"경로가 없습니다: {root}")

    print(f"검사 경로: {root.resolve()}")
    sequences = discover_sequences(
        str(root), fmt=args.format, min_len=args.min_len,
        max_objects=args.max_objects or None, min_visibility=args.min_visibility,
    )

    if not sequences:
        print("\n[실패] 시퀀스를 하나도 찾지 못했습니다.\n")
        print("지원하는 배치 형태:")
        print("  OTB/LaSOT  : <seq>/img/0001.jpg          + <seq>/groundtruth_rect.txt")
        print("  GOT-10k    : <seq>/00000001.jpg          + <seq>/groundtruth.txt")
        print("  MOT        : <seq>/img1/000001.jpg       + <seq>/gt/gt.txt")
        print("  video      : clip.mp4                    + clip.txt")
        print("\n디렉터리 안에 실제로 무엇이 있는지:")
        for p in sorted(root.rglob("*"))[:25]:
            print("   ", p.relative_to(root), "/" if p.is_dir() else "")
        raise SystemExit(1)

    total_frames = sum(len(s) for s in sequences)
    src = Counter("video" if s.video_path else "images" for s in sequences)
    print(f"\n[성공] 시퀀스 {len(sequences)}개 / 주석 프레임 {total_frames}개")
    print(f"       소스: {dict(src)}")

    problems = []
    n_invisible = 0
    for seq in sequences:
        finite = np.isfinite(seq.gt).all(axis=1)
        n_invisible += int((~finite).sum())
        if not finite.any():
            problems.append(f"{seq.uid}: 유효한 GT 박스가 없습니다")
            continue

        if args.check_frames:
            idxs = np.linspace(0, len(seq) - 1, args.check_frames).astype(int)
            for i in idxs:
                img = seq.read(int(i))
                if img is None:
                    problems.append(f"{seq.uid}: {i}번 프레임을 읽지 못했습니다")
                    break
                box = seq.gt[int(i)]
                if not np.all(np.isfinite(box)):
                    continue
                h, w = img.shape[:2]
                if box[0] < -w or box[1] < -h or box[0] > 2 * w or box[1] > 2 * h:
                    problems.append(
                        f"{seq.uid}: {i}번 GT {box.tolist()} 가 이미지 {w}x{h} 와 "
                        f"전혀 맞지 않습니다 (GT 파일이 다른 시퀀스의 것일 수 있음)"
                    )
                    break
        seq.close()

    if n_invisible:
        print(f"       가림/화면이탈로 채점 제외되는 프레임: {n_invisible}개")

    if args.list:
        print("\n시퀀스 목록:")
        for s in sequences:
            print(f"  {s.uid:40s} {len(s):6d} frames")
    else:
        print("\n앞쪽 10개:")
        for s in sequences[:10]:
            print(f"  {s.uid:40s} {len(s):6d} frames")
        if len(sequences) > 10:
            print(f"  ... 외 {len(sequences) - 10}개 (--list 로 전체 보기)")

    if problems:
        print(f"\n[경고] 문제 {len(problems)}건:")
        for msg in problems[:20]:
            print("  -", msg)
        raise SystemExit(1)

    print("\n이상 없습니다. 아래 명령으로 1단계를 시작할 수 있습니다:")
    print(f"  python tools/tune_csrt.py --data-root {root} --output outputs/csrt_labels")


if __name__ == "__main__":
    main()
