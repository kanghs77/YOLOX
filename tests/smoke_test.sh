#!/usr/bin/env bash
# End-to-end integration check of all three stages on synthetic data.
#
#   bash tests/smoke_test.sh [workdir]
#
# This verifies that the code *runs* and that the pieces fit together; it uses a
# randomly-initialised YOLOX-m checkpoint, so the accuracy numbers it prints are
# meaningless. Point --yolox-ckpt at the real yolox_m.pth for a real run.
set -euo pipefail

WORK="${1:-/tmp/csrt_mlp_smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$WORK"

echo "=== synthetic dataset"
python tests/make_synthetic_data.py --output "$WORK/data"

echo "=== stage 1: tune CSRT per scene"
python tools/tune_csrt.py \
    --data-root "$WORK/data" --output "$WORK/labels" \
    --granularity chunk --chunk-size 60 --min-chunk 25 \
    --n-trials 4 --refine-trials 2 --max-frames 30 \
    --workers 4 --dump-frames

echo "=== random-weight YOLOX-m checkpoint (official key layout)"
python - "$WORK/fake_yolox_m.pth" <<'PY'
import sys, torch
sys.path.insert(0, ".")
from csrt_mlp.yolox_min import YOLOPAFPN

model = YOLOPAFPN(0.67, 0.75)                      # yolox-m depth/width
state = {f"backbone.{k}": v for k, v in model.state_dict().items()}
state["head.cls_preds.0.weight"] = torch.zeros(1)  # must be dropped by the loader
torch.save({"model": state, "start_epoch": 0}, sys.argv[1])
print(f"backbone params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
PY

echo "=== stage 2: train the MLP on frozen features"
python tools/train_mlp.py \
    --labels "$WORK/labels/labels.jsonl" \
    --yolox-ckpt "$WORK/fake_yolox_m.pth" --output "$WORK/mlp" \
    --epochs 3 --batch-size 4 --frame-stride 20 --input-size 320 \
    --hidden 256 128 --num-workers 2 --val-ratio 0.25 \
    --cache-dir "$WORK/featcache"

echo "=== stage 2b: rerun, must hit the feature cache"
python tools/train_mlp.py \
    --labels "$WORK/labels/labels.jsonl" \
    --yolox-ckpt "$WORK/fake_yolox_m.pth" --output "$WORK/mlp2" \
    --epochs 2 --batch-size 4 --frame-stride 20 --input-size 320 \
    --hidden 256 128 --num-workers 2 --val-ratio 0.25 \
    --cache-dir "$WORK/featcache" --weight-by-gain

echo "=== stage 3: evaluate"
python tools/eval_tracker.py \
    --data-root "$WORK/data/otb" --mlp-ckpt "$WORK/mlp/best.pth" \
    --tuned-params "$WORK/labels/params_per_unit.json" \
    --frame-stride 2 --output "$WORK/eval.json"

python tools/eval_tracker.py \
    --data-root "$WORK/data/otb" --seqs scene_a \
    --mlp-ckpt "$WORK/mlp/best.pth" --predict-every 40 --frame-stride 2

echo "=== ALL STAGES PASSED"
