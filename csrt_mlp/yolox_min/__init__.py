"""Self-contained subset of Megvii's YOLOX needed to run a frozen YOLOX-m neck."""

from .darknet import CSPDarknet
from .yolo_pafpn import YOLOPAFPN

# (depth, width) of the official YOLOX variants
YOLOX_SIZES = {
    "yolox-nano": (0.33, 0.25),
    "yolox-tiny": (0.33, 0.375),
    "yolox-s": (0.33, 0.50),
    "yolox-m": (0.67, 0.75),
    "yolox-l": (1.00, 1.00),
    "yolox-x": (1.33, 1.25),
}

__all__ = ["CSPDarknet", "YOLOPAFPN", "YOLOX_SIZES"]
