# Scene-adaptive CSRT tuning with a frozen YOLOX-m + MLP

CSRT is a strong but very parameter-sensitive tracker: the settings that work on
a small, fast, low-contrast target are not the ones that work on a large,
slow-moving one. This repo learns that mapping.

1. **Stage 1 - tune CSRT per scene.** For every video with ground-truth boxes,
   search the CSRT hyper-parameters that maximise tracking accuracy on that
   scene, and write the winning parameters out **for every frame**.
2. **Stage 2 - train an MLP.** YOLOX-m is loaded from the stock `yolox_m.pth`,
   **frozen**, and used as a feature extractor. The FPN feature maps at the
   tracked box feed an MLP that regresses the stage-1 parameters.
3. **Stage 3 - evaluate.** Compare default CSRT vs. the stage-1 oracle vs. the
   MLP prediction on held-out clips.

At inference you no longer need ground truth: one forward pass of the frozen
backbone on the init frame gives you the CSRT parameters for that scene.

```
video + GT ──► [stage 1] CSRT search ──► per-frame parameter labels
                                                   │
image ──► [frozen YOLOX-m] ──► FPN maps ──► RoI + context + box geometry
                                                   │
                                                   ▼
                                                  MLP ──► 15 CSRT parameters
```

## Install

```bash
pip install -r requirements.txt
```

`opencv-contrib-python` is required — the plain `opencv-python` wheel ships no
`TrackerCSRT`. Download the official YOLOX-m weights
(`yolox_m.pth`) from the [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases)
into `weights/`.

The YOLOX-m backbone + PAFPN are vendored under `csrt_mlp/yolox_min/` with
upstream-identical module names, so the official checkpoint loads directly and
**no YOLOX installation is needed**. If the real `yolox` package happens to be
importable, it is used instead.

## Data layouts

Auto-detected by `--format auto`:

| Format | Layout |
| --- | --- |
| `otb` (OTB/LaSOT/GOT-10k/UAV123) | `<seq>/img/*.jpg` + `<seq>/groundtruth_rect.txt` (`x,y,w,h` per line) |
| `mot` (MOT16/17/20) | `<seq>/img1/*.jpg` + `<seq>/gt/gt.txt`; every identity becomes one tracking sequence |
| `video` | `clip.mp4` + `clip.txt` (`x,y,w,h` per line) |

VOT 8-value polygon annotations are converted to axis-aligned boxes. Frames
whose GT is missing/`NaN` are skipped rather than scored.

## Stage 1 — tune CSRT per scene

```bash
python tools/tune_csrt.py \
    --data-root /data/tracking \
    --output outputs/csrt_labels \
    --granularity chunk --chunk-size 120 \
    --n-trials 48 --refine-trials 24 \
    --max-frames 300 --workers 8 --dump-frames
```

**What is searched.** 15 numeric CSRT parameters (`csrt_mlp/params_spec.py`):
`padding`, `template_size`, `gsl_sigma`, `filter_lr`, `weights_lr`,
`admm_iterations`, `psr_threshold`, `num_hog_channels_used`, `hog_clip`,
`histogram_lr`, `background_ratio`, `number_of_scales`, `scale_lr`,
`scale_step`, `scale_sigma_factor`. Learning-rate-like parameters are sampled
in log space; boolean switches stay at the OpenCV defaults so the regression
target is continuous.

**How.** Random search (or Optuna TPE with `--sampler tpe`) followed by
Gaussian local refinement with a shrinking radius. The objective is

```
score = mean_IoU − failure_penalty × (resets / scored_frames)
```

with VOT-style resets from ground truth below `--reinit-iou`. The OpenCV
defaults are always evaluated first; if the search cannot beat them by
`--min-gain` the defaults are kept as the label (`used_default: true`), so the
MLP is never taught to move away from a setting that was already fine.

**Granularity.** `--granularity video` gives one parameter set per clip.
`--granularity chunk` tunes each `--chunk-size` window separately, so labels
vary *within* a video — more supervision, and it lets the MLP react to a scene
changing mid-clip. Costs proportionally more tuning time.

**Outputs.**

| File | Contents |
| --- | --- |
| `labels.jsonl` | one record per annotated frame: image/video ref, box, the tuned parameters, their normalised vector, and the achieved score |
| `params_per_unit.json` | tuned parameters per scene/chunk |
| `params_spec.json` | the exact search space used (stages 2/3 decode with it) |
| `summary.json` | default vs. tuned accuracy, overall and per sequence |

`--sidecar` additionally drops a `<image>.csrt.json` next to every frame.
`--dump-frames` decodes video-backed clips to JPEG so stage 2 reads images
directly instead of seeking the container each time.

## Stage 2 — train the MLP on frozen YOLOX-m features

```bash
python tools/train_mlp.py \
    --labels outputs/csrt_labels/labels.jsonl \
    --yolox-ckpt weights/yolox_m.pth \
    --output outputs/mlp \
    --epochs 30 --batch-size 32 --frame-stride 2 \
    --cache-dir outputs/feat_cache --device cuda --amp
```

**Frozen backbone.** `FrozenYoloxFeatures` loads `yolox_m.pth` into a
YOLOPAFPN, drops the detection head, calls `requires_grad_(False)` on every
tensor and pins the module to `eval()` mode — `.train()` is overridden so a
trainer cannot accidentally un-freeze BatchNorm. Preprocessing matches YOLOX's
`ValTransform` (letterbox to 640 with 114 padding, BGR, **no** `/255`).

**The descriptor** (per FPN level `p3`/`p4`/`p5`, strides 8/16/32):

* `roi_size × roi_size` RoI-aligned features at the tracked box → *what is being tracked*
* the globally average-pooled map → *what the scene looks like*
* plus 8 box-geometry features (position, size, log aspect, area)

Target size and scene texture are precisely what drives `template_size`,
`padding` and the `scale_*` group, so both halves earn their place. Use
`--levels`, `--roi-size`, `--no-context`, `--no-geometry` to ablate.

**The head.** LayerNorm → 3 GELU blocks → sigmoid, predicting each parameter in
`[0, 1]`. The sigmoid guarantees a decoded parameter can never leave the tuned
search range, so predictions always construct a valid tracker. Loss is a
weighted Smooth-L1 in normalised space; `--weight-by-gain` down-weights frames
whose tuning barely beat the defaults (i.e. noisy labels).

**Splitting** is by *sequence*, never by frame — neighbouring frames of one clip
are near-duplicates, so a frame-level split would leak badly and report a
meaningless validation number. `--frame-stride 2` drops that redundancy in
training too.

**Caching.** The frozen features never change between epochs, so `--cache-dir`
stores them as float16 on first use and later epochs skip the backbone
entirely. The cache key covers the backbone config, so changing `--levels` or
`--roi-size` will not reuse stale vectors.

## Stage 3 — evaluate

```bash
python tools/eval_tracker.py \
    --data-root /data/tracking \
    --mlp-ckpt outputs/mlp/best.pth \
    --tuned-params outputs/csrt_labels/params_per_unit.json \
    --seqs <the validation clips> \
    --output outputs/eval.json
```

Reports mean IoU / success rate for **default**, **oracle** (stage-1 result on
that clip — the upper bound the MLP is chasing) and **mlp**. Evaluate on the
sequences that were *held out* in stage 2, or the numbers are meaningless.
`--predict-every N` refreshes the prediction every N frames instead of only at
initialisation.

## Using the predictor in your own code

```python
from csrt_mlp.predictor import CSRTParamPredictor
from csrt_mlp.csrt_utils import create_csrt

predictor = CSRTParamPredictor("outputs/mlp/best.pth", device="cuda")
params = predictor.predict(first_frame_bgr, init_box_xywh)   # dict of 15 values
tracker = create_csrt(params)
tracker.init(first_frame_bgr, init_box_xywh)
```

## Layout

```
csrt_mlp/
  params_spec.py    search space + normalise/denormalise (shared by all stages)
  csrt_utils.py     OpenCV-version-tolerant CSRT factory, IoU, tracking evaluation
  datasets.py       OTB / MOT / video+GT discovery and frame access
  tuning.py         random + TPE search with local refinement
  features.py       frozen YOLOX-m extractor, RoI align, preprocessing
  model.py          the MLP head
  train_utils.py    label dataset, sequence-level split, feature cache
  predictor.py      checkpoint -> CSRT parameters
  yolox_min/        vendored YOLOX-m backbone (CSPDarknet + YOLOPAFPN)
tools/
  tune_csrt.py      stage 1
  train_mlp.py      stage 2
  eval_tracker.py   stage 3
```

## Practical notes

* **Tuning cost dominates.** Each trial runs CSRT over the unit's frames.
  `--max-frames`, `--frame-stride` and `--workers` are the knobs; start with
  few clips and `--n-trials 16` to size a full run.
* **Labels are only as good as the search.** If `summary.json` shows a small
  gap between `baseline_mean_iou` and `tuned_mean_iou`, there is little signal
  to learn — raise `--n-trials` before blaming the MLP.
* **Match the tuning objective to your evaluation protocol.** Stage 1 scores a
  *subsample* (`--max-frames`, `--frame-stride`) and resets from ground truth
  (`--reinit-iou`), while stage 3 reports full one-pass IoU by default. With a
  small trial budget the search can overfit that subsample, and the "oracle"
  row then loses to the defaults on the same clip. If that happens, raise
  `--max-frames`/`--n-trials`, or tune with the settings you intend to report.
* **`number_of_scales` is forced odd** on decode, and integer parameters are
  rounded, so every prediction is directly usable by OpenCV.

## Verifying an install

```bash
bash tests/smoke_test.sh /tmp/csrt_mlp_smoke
```

Builds a synthetic OTB/MOT/video dataset and runs all three stages, including
the feature cache and the checkpoint loader. It uses *randomly initialised*
YOLOX-m weights, so it checks that the pipeline runs — not that it is
accurate.
