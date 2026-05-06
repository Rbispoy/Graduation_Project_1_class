"""
图文联合检索时的自适应融合权重 α（答辩创新点 B）
============================================

思路（无额外训练、可解释）：
- **文本越长**：查询语义越具体，略提高文本权重 → **降低 α**（α 表示图像权重）。
- **图像越清晰**（边缘能量近似）：略提高图像权重 → **略提高 α**。

最终 α 限制在 [0.15, 0.85]，避免某一模态被完全压死。
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _text_length_alpha(text: str) -> float:
    n = len(text.strip())
    if n <= 8:
        return 0.78
    if n >= 96:
        return 0.32
    return 0.78 - (n - 8) / (96 - 8) * (0.78 - 0.32)


def _image_clarity_boost(image: Image.Image) -> float:
    """返回约 [-0.12, 0.12] 的微调量。"""
    g = np.asarray(image.convert("L").resize((160, 160)), dtype=np.float32)
    if g.size == 0:
        return 0.0
    dv = float(np.abs(g[1:, :] - g[:-1, :]).mean() + np.abs(g[:, 1:] - g[:, :-1]).mean())
    clarity = min(1.0, dv / 28.0)
    return 0.12 * (clarity - 0.5)


def compute_adaptive_alpha(text: str, image: Image.Image) -> float:
    base = _text_length_alpha(text)
    boost = _image_clarity_boost(image)
    alpha = base + boost
    return float(max(0.15, min(0.85, alpha)))
