"""Discovery / loading of tracking datasets (video or image folder + GT boxes).

Supported layouts (auto-detected):

1. ``otb``   - OTB / LaSOT / GOT-10k / UAV123 style::

       <seq>/img/0001.jpg ...            (or imgs/, color/, frames/, images/)
       <seq>/groundtruth_rect.txt        (or groundtruth.txt, gt.txt)

   One line per frame: ``x,y,w,h`` (comma, space or tab separated).

2. ``mot``   - MOT16/17/20 style::

       <seq>/img1/000001.jpg ...
       <seq>/gt/gt.txt        -> frame,id,x,y,w,h,conf,class,visibility

   Each ground-truth identity becomes its own single-object tracking sequence.

3. ``video`` - a video file plus a sibling GT text file::

       clips/foo.mp4  +  clips/foo.txt   (or foo_gt.txt, annotations/foo.txt)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

__all__ = ["Sequence", "discover_sequences", "load_gt_file", "dump_video_frames"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg")
IMG_DIR_NAMES = ("img", "imgs", "images", "img1", "color", "frames")
GT_FILE_NAMES = (
    "groundtruth_rect.txt", "groundtruth.txt", "groundtruth_rect.1.txt",
    "gt.txt", "groundtruth_rect.1.txt", "anno.txt",
)


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
class Sequence:
    """A single-object tracking sequence: N frames + N ground-truth boxes."""

    def __init__(
        self,
        name: str,
        gt: np.ndarray,
        frame_paths: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        video_indices: Optional[List[int]] = None,
        track_id: int = 0,
    ):
        if frame_paths is None and video_path is None:
            raise ValueError("either frame_paths or video_path must be given")
        self.name = name
        self.track_id = int(track_id)
        self.gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        self.frame_paths = list(frame_paths) if frame_paths else None
        self.video_path = video_path
        # absolute frame numbers inside the video (defaults to 0..N-1)
        self.video_indices = (
            list(video_indices) if video_indices is not None
            else (list(range(len(self.gt))) if video_path else None)
        )
        self._cap = None
        self._cap_pos = -1

        n = len(self.gt)
        if self.frame_paths is not None:
            n = min(n, len(self.frame_paths))
            self.frame_paths = self.frame_paths[:n]
        elif self.video_indices is not None:
            n = min(n, len(self.video_indices))
            self.video_indices = self.video_indices[:n]
        self.gt = self.gt[:n]

    # -- basics -----------------------------------------------------------
    @property
    def uid(self) -> str:
        return f"{self.name}#{self.track_id}"

    def __len__(self) -> int:
        return len(self.gt)

    def __repr__(self) -> str:  # pragma: no cover
        src = "video" if self.video_path else "images"
        return f"<Sequence {self.uid} ({src}, {len(self)} frames)>"

    # -- frame access ------------------------------------------------------
    def read(self, index: int) -> Optional[np.ndarray]:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to read frames")
        if self.frame_paths is not None:
            return cv2.imread(self.frame_paths[index], cv2.IMREAD_COLOR)

        frame_no = self.video_indices[index]
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            self._cap_pos = -1
        # sequential reads are cheap; jumping backwards needs a real seek
        if frame_no != self._cap_pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
        ok, img = self._cap.read()
        self._cap_pos = frame_no if ok else -1
        return img if ok else None

    def frame_ref(self, index: int) -> Dict:
        """Serialisable pointer to one frame (goes into labels.jsonl)."""
        if self.frame_paths is not None:
            return {"image": os.path.abspath(self.frame_paths[index]),
                    "video": None, "frame_no": index}
        return {"image": None, "video": os.path.abspath(self.video_path),
                "frame_no": int(self.video_indices[index])}

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # picklable for multiprocessing (VideoCapture is not)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cap"] = None
        state["_cap_pos"] = -1
        return state


# ---------------------------------------------------------------------------
# ground-truth parsing
# ---------------------------------------------------------------------------
def load_gt_file(path: str) -> np.ndarray:
    """Parse an ``x,y,w,h`` per-line ground-truth file (N, 4)."""
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[,\s;]+", line) if p != ""]
            vals = []
            for p in parts[:8]:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(float("nan"))
            if len(vals) >= 8:
                # polygon (VOT style) -> axis aligned bounding box
                xs, ys = vals[0:8:2], vals[1:8:2]
                vals = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            if len(vals) < 4:
                rows.append([float("nan")] * 4)
            else:
                rows.append(vals[:4])
    if not rows:
        raise ValueError(f"no ground-truth rows parsed from {path}")
    return np.asarray(rows, dtype=np.float64)


def _load_absence_flags(seq_dir: Path, n: int) -> Optional[np.ndarray]:
    """Per-frame "target not visible" flags, if the dataset ships them.

    LaSOT has ``full_occlusion.txt`` / ``out_of_view.txt``, GOT-10k has
    ``absence.label``.  Frames marked absent are blanked to NaN so the tuner
    neither scores them nor re-initialises on them.
    """
    flags = np.zeros(n, dtype=bool)
    found = False
    for fname in ("full_occlusion.txt", "out_of_view.txt", "absence.label"):
        path = seq_dir / fname
        if not path.is_file():
            continue
        try:
            raw = path.read_text(errors="ignore")
            vals = [v for v in re.split(r"[,\s]+", raw.strip()) if v != ""]
            arr = np.array([float(v) != 0 for v in vals], dtype=bool)
        except (ValueError, OSError):
            continue
        if arr.size == 0:
            continue
        found = True
        m = min(n, arr.size)
        flags[:m] |= arr[:m]
    return flags if found else None


def _apply_absence(seq_dir: Path, gt: np.ndarray) -> np.ndarray:
    flags = _load_absence_flags(seq_dir, len(gt))
    if flags is not None:
        gt = gt.copy()
        gt[flags] = np.nan
    return gt


def _list_images(d: Path) -> List[str]:
    files = [str(p) for p in sorted(d.iterdir())
             if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return files


def _find_img_dir(seq_dir: Path) -> Optional[Path]:
    for name in IMG_DIR_NAMES:
        cand = seq_dir / name
        if cand.is_dir() and _list_images(cand):
            return cand
    # sequence folder that directly holds the frames
    if _list_images(seq_dir):
        return seq_dir
    return None


def _find_gt_file(seq_dir: Path) -> Optional[Path]:
    for name in GT_FILE_NAMES:
        cand = seq_dir / name
        if cand.is_file():
            return cand
    txts = sorted(p for p in seq_dir.glob("*.txt") if p.is_file())
    return txts[0] if txts else None


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def _load_otb_sequence(seq_dir: Path, min_len: int) -> List[Sequence]:
    img_dir = _find_img_dir(seq_dir)
    if img_dir is None:
        return []
    frames = _list_images(img_dir)
    if not frames:
        return []

    # OTB's Jogging / Skating2 annotate two targets in one folder
    multi = sorted(seq_dir.glob("groundtruth_rect.[0-9].txt"))
    gt_files = multi if multi else ([f] if (f := _find_gt_file(seq_dir)) else [])
    if not gt_files:
        return []

    out: List[Sequence] = []
    for track_id, gt_file in enumerate(gt_files):
        try:
            gt = load_gt_file(str(gt_file))
        except (ValueError, OSError):
            continue
        n = min(len(frames), len(gt))
        if n < min_len:
            continue
        out.append(Sequence(seq_dir.name, _apply_absence(seq_dir, gt[:n]),
                            frame_paths=frames[:n], track_id=track_id))
    return out


def _load_mot_sequence(
    seq_dir: Path, min_len: int, classes=(1, 2, 7), min_visibility: float = 0.3,
    max_objects: Optional[int] = None,
) -> List[Sequence]:
    img_dir = seq_dir / "img1"
    gt_file = seq_dir / "gt" / "gt.txt"
    if not img_dir.is_dir() or not gt_file.is_file():
        return []
    frames = _list_images(img_dir)
    if not frames:
        return []

    per_id: Dict[int, Dict[int, List[float]]] = {}
    with open(gt_file, "r", errors="ignore") as fh:
        for line in fh:
            parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
            if len(parts) < 6:
                continue
            try:
                fr, tid = int(float(parts[0])), int(float(parts[1]))
                x, y, w, h = (float(v) for v in parts[2:6])
            except ValueError:
                continue
            if len(parts) >= 7 and float(parts[6]) == 0:
                continue                                    # ignored detection
            if len(parts) >= 8 and classes and int(float(parts[7])) not in classes:
                continue
            if len(parts) >= 9 and float(parts[8]) < min_visibility:
                continue
            per_id.setdefault(tid, {})[fr] = [x, y, w, h]

    sequences: List[Sequence] = []
    for tid, boxes in sorted(per_id.items()):
        # keep the longest contiguous run of frames for this identity
        fr_sorted = sorted(boxes)
        best, cur = [], [fr_sorted[0]]
        for prev, nxt in zip(fr_sorted, fr_sorted[1:]):
            if nxt == prev + 1:
                cur.append(nxt)
            else:
                best, cur = (cur if len(cur) > len(best) else best), [nxt]
        best = cur if len(cur) > len(best) else best
        if len(best) < min_len:
            continue
        paths, gts = [], []
        for fr in best:
            if 1 <= fr <= len(frames):
                paths.append(frames[fr - 1])       # MOT frame numbers are 1-based
                gts.append(boxes[fr])
        if len(paths) < min_len:
            continue
        sequences.append(Sequence(seq_dir.name, np.asarray(gts), frame_paths=paths,
                                  track_id=tid))

    sequences.sort(key=len, reverse=True)
    return sequences[:max_objects] if max_objects else sequences


def _find_video_gt(video: Path) -> Optional[Path]:
    stem = video.stem
    cands = [
        video.with_suffix(".txt"),
        video.parent / f"{stem}_gt.txt",
        video.parent / f"{stem}_groundtruth.txt",
        video.parent / "annotations" / f"{stem}.txt",
        video.parent.parent / "annotations" / f"{stem}.txt",
        video.parent / stem / "groundtruth_rect.txt",
    ]
    for c in cands:
        if c.is_file():
            return c
    return None


def _load_video_sequence(video: Path, min_len: int) -> List[Sequence]:
    gt_file = _find_video_gt(video)
    if gt_file is None:
        return []
    gt = load_gt_file(str(gt_file))
    if len(gt) < min_len:
        return []
    return [Sequence(video.stem, gt, video_path=str(video))]


def discover_sequences(
    root: str,
    fmt: str = "auto",
    min_len: int = 20,
    max_objects: Optional[int] = None,
    mot_classes=(1, 2, 7),
    min_visibility: float = 0.3,
) -> List[Sequence]:
    """Walk ``root`` and build every sequence it can find."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root)

    sequences: List[Sequence] = []

    # a single sequence folder / single video passed directly
    if root_path.is_file() and root_path.suffix.lower() in VIDEO_EXTS:
        return _load_video_sequence(root_path, min_len)

    candidates = [root_path] + [p for p in sorted(root_path.rglob("*")) if p.is_dir()]
    seen = set()
    for d in candidates:
        if str(d) in seen:
            continue
        seen.add(str(d))
        found: List[Sequence] = []
        if fmt in ("auto", "mot"):
            found = _load_mot_sequence(d, min_len, mot_classes, min_visibility, max_objects)
        if not found and fmt in ("auto", "otb"):
            found = _load_otb_sequence(d, min_len)
        if found:
            sequences.extend(found)

    if fmt in ("auto", "video"):
        for p in sorted(root_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                sequences.extend(_load_video_sequence(p, min_len))

    # de-duplicate (a nested folder can be picked up twice)
    uniq: Dict[str, Sequence] = {}
    for s in sequences:
        uniq.setdefault(s.uid, s)
    out = sorted(uniq.values(), key=lambda s: s.uid)
    return out


# ---------------------------------------------------------------------------
# optional: materialise video frames so stage 2 can read them as images
# ---------------------------------------------------------------------------
def dump_video_frames(seq: Sequence, out_dir: str, quality: int = 95) -> Sequence:
    """Decode a video-backed sequence to JPEGs and return an image-backed one."""
    if seq.video_path is None:
        return seq
    if cv2 is None:
        raise RuntimeError("OpenCV is required to dump frames")
    dst = Path(out_dir) / seq.uid.replace("#", "_")
    dst.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(len(seq)):
        p = dst / f"{i:06d}.jpg"
        if not p.exists():
            img = seq.read(i)
            if img is None:
                break
            cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        paths.append(str(p))
    seq.close()
    n = len(paths)
    return Sequence(seq.name, seq.gt[:n], frame_paths=paths, track_id=seq.track_id)
