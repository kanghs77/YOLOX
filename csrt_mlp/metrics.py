"""추적 평가 지표 (OTB 표준) 와 전략 간 비교 리포트 생성."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence

import numpy as np

__all__ = ["success_curve", "precision_curve", "summarize_run",
           "aggregate", "format_table", "to_csv"]

def _width(text: str) -> int:
    """터미널 표시 폭. 한글/전각 문자는 2칸을 차지한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, n: int, align: str = "<") -> str:
    """표시 폭 기준 정렬 (한글 헤더가 섞여도 표가 어긋나지 않게)."""
    fill = max(0, n - _width(text))
    if align == ">":
        return " " * fill + text
    return text + " " * fill


# OTB 관례: IoU 임계값 0:0.05:1, 중심오차 임계값 0~50px
SUCCESS_THRESHOLDS = np.arange(0.0, 1.0 + 1e-9, 0.05)
PRECISION_THRESHOLDS = np.arange(0.0, 50.0 + 1e-9, 1.0)


def success_curve(ious: Sequence[float]) -> np.ndarray:
    """임계값별 success rate. 면적(AUC)이 곧 success score."""
    if not len(ious):
        return np.zeros_like(SUCCESS_THRESHOLDS)
    arr = np.asarray(ious, dtype=np.float64)
    return np.array([(arr > t).mean() for t in SUCCESS_THRESHOLDS])


def precision_curve(errors: Sequence[float]) -> np.ndarray:
    """중심오차 임계값별 precision."""
    if not len(errors):
        return np.zeros_like(PRECISION_THRESHOLDS)
    arr = np.asarray(errors, dtype=np.float64)
    return np.array([(arr <= t).mean() for t in PRECISION_THRESHOLDS])


def summarize_run(result: Dict) -> Dict[str, float]:
    """``track_sequence`` 결과 하나를 지표 묶음으로 변환."""
    ious = result.get("ious", [])
    errors = result.get("center_errors", [])
    sc = success_curve(ious)
    pc = precision_curve(errors)
    return {
        "mean_iou": float(result.get("mean_iou", 0.0)),
        # AUC: 임계값에 대한 평균 = 곡선 아래 면적 (OTB 의 success score)
        "success_auc": float(sc.mean()),
        "success_50": float(np.mean([i > 0.5 for i in ious])) if len(ious) else 0.0,
        # precision plot 의 관례적 대표값은 20px
        "precision_20": float(pc[20]) if len(pc) > 20 else 0.0,
        "n_failures": int(result.get("n_failures", 0)),
        "n_eval": int(result.get("n_eval", 0)),
        "fps": float(result.get("fps", 0.0)),
        "score": float(result.get("score", 0.0)),
    }


# 표에 싣는 지표와 표시 이름 (높을수록 좋음)
METRIC_LABELS = [
    ("mean_iou", "mean IoU"),
    ("success_auc", "Success AUC"),
    ("success_50", "Success@0.5"),
    ("precision_20", "Prec@20px"),
    ("fps", "FPS"),
]


def aggregate(rows: List[Dict], strategies: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """시퀀스별 결과를 전략별 평균으로 집계 (시퀀스 단위 평균)."""
    out: Dict[str, Dict[str, float]] = {}
    for name in strategies:
        vals = [r[name] for r in rows if name in r]
        if not vals:
            continue
        out[name] = {
            key: float(np.mean([v[key] for v in vals]))
            for key, _ in METRIC_LABELS
        }
        out[name]["n_failures"] = float(np.sum([v["n_failures"] for v in vals]))
        out[name]["n_sequences"] = float(len(vals))
    return out


def format_table(rows: List[Dict], summary: Dict[str, Dict[str, float]],
                 strategies: Sequence[str], baseline: str = "default") -> str:
    """사람이 읽는 비교표 + 기준 대비 승패 집계."""
    present = [s for s in strategies if s in summary]
    lines: List[str] = []

    # --- 전체 요약 ---------------------------------------------------
    header = _pad("전략", 10) + "".join(_pad(label, 13, ">") for _, label in METRIC_LABELS)
    rule = "=" * _width(header)
    lines.append(rule)
    lines.append("전체 요약 (시퀀스 평균)")
    lines.append(rule)
    lines.append(header)
    lines.append("-" * _width(header))
    for name in present:
        v = summary[name]
        lines.append(_pad(name, 10) + "".join(f"{v[k]:13.4f}" for k, _ in METRIC_LABELS))

    # --- 기준 대비 증감 -----------------------------------------------
    if baseline in summary:
        lines.append("-" * _width(header))
        for name in present:
            if name == baseline:
                continue
            v, b = summary[name], summary[baseline]
            deltas = "".join(f"{v[k] - b[k]:+13.4f}" for k, _ in METRIC_LABELS)
            lines.append(_pad("Δ " + name, 10) + deltas)

    # --- 시퀀스별 승패 -------------------------------------------------
    if baseline in summary:
        lines.append("")
        lines.append(f"시퀀스별 mean IoU 기준 '{baseline}' 대비 승패 (동률 = ±0.005 이내)")
        for name in present:
            if name == baseline:
                continue
            win = tie = loss = 0
            for r in rows:
                if name not in r or baseline not in r:
                    continue
                d = r[name]["mean_iou"] - r[baseline]["mean_iou"]
                if d > 0.005:
                    win += 1
                elif d < -0.005:
                    loss += 1
                else:
                    tie += 1
            total = win + tie + loss
            rate = (win / total * 100) if total else 0.0
            lines.append(f"  {_pad(name, 10)} 승 {win:3d} / 무 {tie:3d} / 패 {loss:3d}"
                         f"   (승률 {rate:.1f}%)")

    # --- 시퀀스별 상세 -------------------------------------------------
    lines.append("")
    lines.append("시퀀스별 mean IoU")
    sub = _pad("시퀀스", 32) + "".join(_pad(s, 12, ">") for s in present)
    if baseline in summary and len(present) > 1:
        sub += _pad("Δ 최고", 12, ">")
    lines.append(sub)
    lines.append("-" * _width(sub))
    for r in sorted(rows, key=lambda x: x["seq"]):
        line = _pad(r["seq"], 32)
        for s in present:
            line += f"{r[s]['mean_iou']:12.4f}" if s in r else f"{'-':>12}"
        if baseline in summary and len(present) > 1 and baseline in r:
            others = [r[s]["mean_iou"] for s in present if s != baseline and s in r]
            if others:
                line += f"{max(others) - r[baseline]['mean_iou']:+12.4f}"
        lines.append(line)
    return "\n".join(lines)


def to_csv(rows: List[Dict], strategies: Sequence[str]) -> str:
    """시퀀스별 전 지표를 CSV 로 (엑셀/논문 표 작성용)."""
    keys = [k for k, _ in METRIC_LABELS] + ["n_failures", "n_eval"]
    head = ["seq", "n_frames"] + [f"{s}_{k}" for s in strategies for k in keys]
    out = [",".join(head)]
    for r in sorted(rows, key=lambda x: x["seq"]):
        cells = [r["seq"], str(r.get("n_frames", ""))]
        for s in strategies:
            for k in keys:
                cells.append(f"{r[s][k]:.6f}" if s in r else "")
        out.append(",".join(cells))
    return "\n".join(out) + "\n"
