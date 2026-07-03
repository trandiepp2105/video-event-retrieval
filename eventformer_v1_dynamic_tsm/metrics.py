from __future__ import annotations

from typing import Tuple


def span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    a_s, a_e = a
    b_s, b_e = b
    inter = max(0, min(a_e, b_e) - max(a_s, b_s) + 1)
    union = max(a_e, b_e) - min(a_s, b_s) + 1
    if union <= 0:
        return 0.0
    return inter / union
