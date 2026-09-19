"""Scene-adaptive CSRT tuning + MLP parameter prediction on frozen YOLOX-m features.

Pipeline
--------
1. ``tools/tune_csrt.py``   - tune CSRT per scene on video+GT data, save the best
   parameters for every frame (``labels.jsonl``).
2. ``tools/train_mlp.py``   - freeze YOLOX-m, take FPN features at the tracked box
   and regress those parameters with an MLP.
3. ``tools/eval_tracker.py`` - compare default / oracle-tuned / MLP-predicted
   parameters on held-out sequences.
"""

__version__ = "0.1.0"

from .params_spec import (  # noqa: F401
    CSRT_SEARCH_SPACE,
    ParamSpec,
    default_params,
    denormalize,
    normalize,
    param_names,
)

__all__ = [
    "CSRT_SEARCH_SPACE",
    "ParamSpec",
    "default_params",
    "denormalize",
    "normalize",
    "param_names",
]
