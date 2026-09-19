# 장면 적응형 CSRT 튜닝 — Frozen YOLOX-m + MLP

CSRT는 성능은 좋지만 파라미터에 매우 민감한 추적기입니다. 작고 빠르며 대비가 낮은
표적에 잘 맞는 설정과, 크고 느린 표적에 잘 맞는 설정은 서로 다릅니다.
이 저장소는 **그 대응 관계를 학습**합니다.

1. **1단계 — 영상별 CSRT 튜닝.** GT가 있는 각 영상에 대해 추적 정확도를 최대화하는
   CSRT 하이퍼파라미터를 탐색하고, 그 결과를 **이미지(프레임)마다** 저장합니다.
2. **2단계 — MLP 학습.** 기본 `yolox_m.pth`를 불러와 **freeze**하고, 추적 박스 위치의
   FPN 특징맵을 MLP에 넣어 1단계에서 저장한 파라미터를 정답으로 회귀 학습합니다.
3. **3단계 — 비교 평가.** 기존 고정 파라미터 CSRT와 MLP 예측 파라미터 CSRT를
   동일 조건에서 비교합니다.

추론 시에는 GT가 필요 없습니다. 초기 프레임 한 장에 대해 frozen 백본을 한 번
forward 하면 그 장면에 맞는 CSRT 파라미터가 나옵니다.

```
영상 + GT ──► [1단계] CSRT 탐색 ──► 프레임별 파라미터 라벨
                                              │
이미지 ──► [frozen YOLOX-m] ──► FPN 특징맵 ──► RoI + 전역맥락 + 박스기하
                                              │
                                              ▼
                                             MLP ──► CSRT 파라미터 15개
```

## 설치

```bash
pip install -r requirements.txt
```

`opencv-contrib-python`이 **반드시** 필요합니다 — 일반 `opencv-python` 휠에는
`TrackerCSRT`가 없습니다.

YOLOX-m 가중치는 [YOLOX 릴리스](https://github.com/Megvii-BaseDetection/YOLOX/releases)에서
`yolox_m.pth`를 받아 `weights/`에 둡니다.

```bash
mkdir -p weights
wget -P weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
```

YOLOX-m의 backbone과 PAFPN은 `csrt_mlp/yolox_min/`에 **upstream과 동일한 모듈명으로
포함**되어 있어, 공식 체크포인트가 그대로 로드되며 **YOLOX를 따로 설치할 필요가
없습니다.** 실제 `yolox` 패키지가 import 가능하면 그쪽을 우선 사용합니다.

---

## 1. 데이터셋을 어디에 두어야 하는가

`--data-root` 아래에 **압축을 푼 그대로** 두면 됩니다. 별도 변환 스크립트는
필요 없습니다. 권장 배치:

```
YOLOX/
├── weights/yolox_m.pth
└── datasets/
    ├── OTB100/
    │   ├── Basketball/
    │   │   ├── img/0001.jpg ...
    │   │   └── groundtruth_rect.txt
    │   └── Jogging/
    │       ├── img/0001.jpg ...
    │       ├── groundtruth_rect.1.txt      ← 표적 2개는 각각 별도 트랙으로 읽힘
    │       └── groundtruth_rect.2.txt
    │
    ├── LaSOT/
    │   └── airplane/
    │       └── airplane-1/
    │           ├── img/00000001.jpg ...
    │           ├── groundtruth.txt
    │           ├── full_occlusion.txt      ← 있으면 자동으로 채점에서 제외
    │           └── out_of_view.txt
    │
    ├── GOT-10k/
    │   └── train/
    │       └── GOT-10k_Train_000001/
    │           ├── 00000001.jpg ...        ← img/ 하위폴더 없이 바로 두어도 됨
    │           ├── groundtruth.txt
    │           └── absence.label           ← 있으면 자동으로 채점에서 제외
    │
    ├── MOT17/
    │   └── train/
    │       └── MOT17-02-FRCNN/
    │           ├── img1/000001.jpg ...
    │           └── gt/gt.txt               ← ID 하나하나가 단일표적 시퀀스가 됨
    │
    └── MOT20/  (MOT17과 동일 구조)
```

### 자동 인식 규칙

| 형식 | 조건 |
| --- | --- |
| `otb` | `<seq>/img/*.jpg` + `groundtruth_rect.txt` 또는 `groundtruth.txt` |
| `otb` | 프레임이 `<seq>/` 바로 아래 있어도 인식 (GOT-10k) |
| `mot` | `<seq>/img1/*.jpg` + `<seq>/gt/gt.txt` |
| `video` | `clip.mp4` + 같은 이름의 `clip.txt` |

- 이미지 폴더명은 `img`, `imgs`, `images`, `img1`, `color`, `frames` 중 아무거나 인식합니다.
- GT는 `x,y,w,h` 한 줄에 하나이며 쉼표/공백/탭 모두 허용합니다.
  VOT식 8개 좌표(폴리곤)는 자동으로 축정렬 박스로 변환됩니다.
- MOT는 `conf=0`인 행, 지정한 클래스가 아닌 행, `visibility < --min-visibility`인
  행을 제외하고, 각 ID의 **가장 긴 연속 구간**을 하나의 시퀀스로 만듭니다.

### 배치가 맞는지 먼저 확인

```bash
python tools/check_dataset.py --data-root datasets/OTB100
```

시퀀스를 실제로 열어 프레임을 읽고 GT 박스가 이미지 범위와 맞는지까지 확인합니다.
못 찾으면 디렉터리에 무엇이 들어 있는지 같이 출력해 주므로 원인을 바로 알 수 있습니다.

### 규모 관련 주의

LaSOT(1,400개)·GOT-10k(9,335개)는 매우 큽니다. 1단계 탐색 비용은
**시퀀스 수 × 트라이얼 수 × 프레임 수**에 비례하므로, 먼저 작게 돌려 시간을
가늠하세요.

```bash
# 20개 시퀀스로 감 잡기
python tools/tune_csrt.py --data-root datasets/LaSOT --output outputs/try \
    --max-sequences 20 --n-trials 16 --max-frames 200 --workers 8
```

> UAV123은 GT가 `anno/` 별도 폴더에 있어 자동 인식되지 않습니다.
> 각 시퀀스 폴더 안에 `groundtruth_rect.txt`로 복사해 두면 `otb` 형식으로 읽힙니다.

---

## 2. MLP는 어느 레이어에 붙는가

**YOLOX-m의 neck(YOLOPAFPN) 출력 3개**, 즉 검출 head 바로 직전 단계에 붙습니다.
검출 head는 아예 로드하지 않고 버립니다.

```
입력 640×640
   │
   ├─ CSPDarknet backbone ── dark3 ─┐  dark4 ─┐  dark5 ─┐
   │                                │         │         │
   └─ YOLOPAFPN (neck) ─────────────┴─────────┴─────────┘
             │
             ├── pan_out2  =  p3   stride  8   192 ch   (80×80)  ← 여기
             ├── pan_out1  =  p4   stride 16   384 ch   (40×40)  ← 여기
             └── pan_out0  =  p5   stride 32   768 ch   (20×20)  ← 여기
                    │
                    │  (YOLOXHead 는 사용하지 않음 — 로드 시 head.* 키는 버림)
                    ▼
        레벨마다:  RoIAlign(3×3, 추적 박스 위치)  →  C×9      "무엇을 추적하는가"
                   Global Average Pooling         →  C        "장면이 어떤가"
                    │
                    ├── + 박스 기하 8차원 (중심 x/y, w/h, log 종횡비, 면적 등)
                    ▼
        concat 13,448 차원
                    ▼
        LayerNorm → Linear 1024 → LN → GELU → Dropout
                  → Linear  512 → LN → GELU → Dropout
                  → Linear  256 → LN → GELU → Dropout
                  → Linear   15 → sigmoid
                    ▼
        정규화된 CSRT 파라미터 15개 ∈ [0,1]  →  역정규화  →  실제 값
```

특징 차원 계산 (yolox-m, width 0.75, `--roi-size 3` 기준):

```
p3: 192×3×3 + 192 = 1,920
p4: 384×3×3 + 384 = 3,840
p5: 768×3×3 + 768 = 7,680
박스 기하                =     8
────────────────────────────────
합계                     = 13,448
```

**왜 이 레이어인가.** PAFPN 출력은 top-down·bottom-up 경로를 모두 거쳐 저수준 질감과
고수준 의미가 함께 섞여 있고, 검출기가 실제로 판단에 쓰는 표현입니다. 또한
stride 8/16/32 세 해상도를 동시에 제공하므로 작은 표적(p3)과 큰 표적(p5)을 모두
커버합니다.

**왜 RoI와 전역 풀링을 같이 쓰는가.** `template_size`·`padding`·`scale_*`를 실제로
좌우하는 것은 *표적의 크기*이고, `filter_lr`·`histogram_lr`은 *장면의 혼잡도/질감*에
좌우됩니다. 전자는 RoI 특징이, 후자는 전역 풀링이 담당합니다.

레이어 선택은 실험으로 바꿀 수 있습니다.

```bash
--levels p3 p4 p5     # 기본값
--levels p4           # 단일 레벨만 (특징 3,848차원, 훨씬 가벼움)
--roi-size 5          # RoI 해상도 ↑
--no-context          # 전역 풀링 제거
--no-geometry         # 박스 기하 제거
```

**freeze 보장.** `requires_grad_(False)`에 더해 `train()`을 오버라이드해 백본을 항상
`eval()`에 고정합니다. 학습 루프가 실수로 BatchNorm 통계를 갱신하는 일이 없습니다.
학습되는 파라미터는 MLP(약 14M)뿐이고 YOLOX-m 21.03M은 동결됩니다.

---

## 3. 1단계 — 영상별 CSRT 튜닝

```bash
python tools/tune_csrt.py \
    --data-root datasets/OTB100 \
    --output outputs/csrt_labels \
    --granularity chunk --chunk-size 120 \
    --n-trials 48 --refine-trials 24 \
    --max-frames 300 --workers 8
```

**탐색 대상** — 수치형 CSRT 파라미터 15개 (`csrt_mlp/params_spec.py`):
`padding`, `template_size`, `gsl_sigma`, `filter_lr`, `weights_lr`,
`admm_iterations`, `psr_threshold`, `num_hog_channels_used`, `hog_clip`,
`histogram_lr`, `background_ratio`, `number_of_scales`, `scale_lr`,
`scale_step`, `scale_sigma_factor`.
학습률 성격의 값은 로그 공간에서 샘플링하고, boolean 스위치는 기본값으로 고정해
회귀 대상이 연속값이 되도록 했습니다.

**탐색 방식** — 랜덤 탐색(또는 `--sampler tpe`로 Optuna TPE) 후, 반경을 줄여 가며
가우시안 국소 정제. 목적함수는

```
score = 평균 IoU − failure_penalty × (재초기화 횟수 / 채점 프레임 수)
```

`--reinit-iou` 미만으로 떨어지면 GT로 재초기화하는 VOT 방식입니다.

**기본값 보호** — OpenCV 기본값을 항상 먼저 평가하고, 탐색이 이를 `--min-gain` 이상
이기지 못하면 기본값을 그대로 라벨로 남깁니다(`used_default: true`).
이미 충분히 좋은 설정에서 벗어나도록 MLP가 학습되는 일을 막습니다.

**granularity** — `video`는 클립당 한 벌, `chunk`는 `--chunk-size` 프레임마다 따로
튜닝해 **영상 내부에서도 라벨이 변합니다.** 지도 신호가 풍부해지고 장면이 중간에
바뀌는 경우에 대응할 수 있지만, 튜닝 시간이 비례해 늘어납니다.

**출력물**

| 파일 | 내용 |
| --- | --- |
| `labels.jsonl` | 주석 프레임마다 1줄: 이미지 경로, 박스, 튜닝된 파라미터, 정규화 벡터, 달성 점수 |
| `params_per_unit.json` | 장면/청크별 튜닝 결과 |
| `params_spec.json` | 사용한 탐색 공간 (2·3단계가 이걸로 역정규화) |
| `summary.json` | 기본값 대비 튜닝 성능, 전체 및 시퀀스별 |

`--sidecar`를 주면 프레임마다 `<이미지경로>.csrt.json`도 함께 저장합니다.
`--dump-frames`는 영상 기반 시퀀스를 JPEG로 풀어 2단계가 컨테이너를 매번 seek 하지
않게 합니다.

---

## 4. 2단계 — Frozen YOLOX-m 특징으로 MLP 학습

```bash
python tools/train_mlp.py \
    --labels outputs/csrt_labels/labels.jsonl \
    --yolox-ckpt weights/yolox_m.pth \
    --output outputs/mlp \
    --epochs 30 --batch-size 32 --frame-stride 2 \
    --cache-dir outputs/feat_cache --device cuda --amp
```

**전처리**는 YOLOX `ValTransform`과 동일합니다 (640 레터박스, 114 패딩, BGR,
`/255` **안 함**).

**손실**은 정규화 공간에서의 가중 Smooth-L1입니다. 출력단 sigmoid 덕분에 역정규화된
파라미터가 탐색 범위를 절대 벗어나지 않아, 예측값을 그대로 `cv2.TrackerCSRT`에
넣을 수 있습니다. 정수형은 반올림되고 `number_of_scales`는 홀수로 강제됩니다.
`--weight-by-gain`을 주면 튜닝 이득이 작았던(= 라벨이 불확실한) 프레임의 가중치를
낮춥니다.

**분할은 프레임이 아니라 시퀀스 단위**입니다. 인접 프레임은 사실상 중복이라
프레임 단위로 나누면 누수가 심해 검증 점수가 무의미해집니다.
학습에 쓴/안 쓴 시퀀스 목록은 `train_sequences.txt`·`val_sequences.txt`로
저장되어 3단계에서 그대로 쓰입니다.

**특징 캐시** — frozen 특징은 epoch 간 변하지 않으므로 `--cache-dir`에 float16으로
저장해 두 번째 epoch부터 백본 forward를 건너뜁니다. 캐시 키에 백본 설정이 포함되어
`--levels`나 `--roi-size`를 바꾸면 이전 벡터를 재사용하지 않습니다.

---

## 5. 3단계 — 고정 파라미터 CSRT vs MLP CSRT 비교

```bash
python tools/eval_tracker.py \
    --data-root datasets/OTB100 \
    --mlp-ckpt outputs/mlp/best.pth \
    --tuned-params outputs/csrt_labels/params_per_unit.json \
    --seqs-file outputs/mlp/val_sequences.txt \
    --output outputs/eval
```

세 전략을 **같은 시퀀스·같은 프로토콜**로 돌려 비교합니다.

| 전략 | 의미 |
| --- | --- |
| `default` | OpenCV 기본값 — 기존 고정 변수 CSRT |
| `mlp` | MLP가 장면을 보고 예측한 파라미터 |
| `oracle` | 1단계가 그 클립의 GT를 보고 찾은 값 — MLP가 목표로 하는 **상한선** |

지표는 OTB 관례를 따릅니다.

- **mean IoU**
- **Success AUC** — IoU 임계값 0:0.05:1 곡선의 면적
- **Success@0.5** — IoU > 0.5 비율
- **Prec@20px** — 중심오차 ≤ 20px 비율
- **FPS** — `tracker.update`만 측정 (프레임 디코딩 제외).
  `number_of_scales`·`admm_iterations`는 속도를 직접 바꾸므로,
  **정확도가 올라가도 속도가 떨어지면 의미가 다릅니다.** 반드시 같이 보세요.

출력은 ① 전체 요약, ② `default` 대비 증감(Δ), ③ 시퀀스별 승/무/패와 승률,
④ 시퀀스별 상세 표 순으로 나오며, `--output`을 주면
`eval.json`·`eval.csv`·`eval.md`로 저장됩니다.

```
전략          mean IoU  Success AUC  Success@0.5    Prec@20px          FPS
----------------------------------------------------------------------------
default         0.5821       0.5604       0.6912       0.7431      41.2314
mlp             0.6147       0.5931       0.7350       0.7812      38.7765
oracle          0.6602       0.6388       0.7904       0.8250      36.1190
----------------------------------------------------------------------------
Δ mlp          +0.0326      +0.0327      +0.0438      +0.0381      -2.4549

시퀀스별 mean IoU 기준 'default' 대비 승패 (동률 = ±0.005 이내)
  mlp        승  38 / 무   9 / 패  13   (승률 63.3%)
```

*(위 숫자는 출력 형식 예시입니다 — 실제 값은 데이터셋과 학습 결과에 따라 다릅니다.)*

**반드시 2단계에서 학습에 쓰이지 않은 시퀀스로 평가하세요.** `--seqs-file`에
`val_sequences.txt`를 주면 자동으로 지켜집니다. 그렇지 않으면 MLP는 이미 정답을 본
클립을 평가하는 셈이라 숫자가 무의미합니다.

`--predict-every N`을 주면 초기 프레임에서 한 번만 예측하는 대신 N 프레임마다 다시
예측합니다.

---

## 코드에서 직접 쓰기

```python
from csrt_mlp.predictor import CSRTParamPredictor
from csrt_mlp.csrt_utils import create_csrt

predictor = CSRTParamPredictor("outputs/mlp/best.pth", device="cuda")
params = predictor.predict(first_frame_bgr, init_box_xywh)   # 15개 값의 dict
tracker = create_csrt(params)
tracker.init(first_frame_bgr, init_box_xywh)

ok, box = tracker.update(next_frame_bgr)
```

## 구조

```
csrt_mlp/
  params_spec.py    탐색 공간 + 정규화/역정규화 (전 단계 공용)
  csrt_utils.py     OpenCV 버전 호환 CSRT 생성, IoU/중심오차, 추적 평가
  datasets.py       OTB / LaSOT / GOT-10k / MOT / video 탐색 및 프레임 접근
  tuning.py         랜덤·TPE 탐색 + 국소 정제
  features.py       frozen YOLOX-m 추출기, RoIAlign, 전처리
  model.py          MLP 헤드
  metrics.py        Success/Precision 곡선, 비교표·CSV 생성
  train_utils.py    라벨 데이터셋, 시퀀스 단위 분할, 특징 캐시
  predictor.py      체크포인트 → CSRT 파라미터
  yolox_min/        YOLOX-m backbone + PAFPN (upstream 동일 모듈명)
tools/
  check_dataset.py  데이터셋 배치 검사
  tune_csrt.py      1단계
  train_mlp.py      2단계
  eval_tracker.py   3단계 (비교)
```

## 설치 검증

```bash
bash tests/smoke_test.sh /tmp/csrt_mlp_smoke
```

합성 데이터셋을 만들어 3단계 전체를 돌립니다(특징 캐시·체크포인트 로더 포함).
**무작위 초기화된** YOLOX-m 가중치를 쓰므로 파이프라인이 도는지만 확인하며,
정확도는 의미가 없습니다.

## 실무 팁

- **튜닝 비용이 지배적입니다.** 트라이얼마다 CSRT를 해당 구간 전체에 돌립니다.
  `--max-frames`, `--frame-stride`, `--workers`가 핵심 조절 손잡이입니다.
- **라벨 품질은 탐색 품질을 넘지 못합니다.** `summary.json`에서
  `baseline_mean_iou`와 `tuned_mean_iou` 차이가 작다면 배울 신호 자체가 적은
  것이므로, MLP를 탓하기 전에 `--n-trials`를 올리세요.
- **튜닝 목적함수와 평가 프로토콜을 맞추세요.** 1단계는 프레임을 부분 샘플링하고
  (`--max-frames`, `--frame-stride`) GT 재초기화(`--reinit-iou`)를 쓰는 반면,
  3단계는 기본이 전체 프레임 one-pass입니다. 트라이얼 예산이 적으면 탐색이 부분
  샘플에 과적합해 **oracle이 기본값보다 나빠질 수도 있습니다.** 그럴 땐
  `--max-frames`/`--n-trials`를 올리거나, 보고할 설정 그대로 튜닝하세요.
