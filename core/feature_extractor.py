"""
Chinese CLIP 特征提取模块
========================

架构说明
--------
本检索引擎使用 **OFA-Sys/chinese-clip-vit-base-patch16**：
- 图像编码器：ViT-B/16，将图片映射到与文本对齐的语义空间。
- 文本编码器：中文 RoBERTa 类结构，支持中文查询。

检索时我们在同一嵌入空间计算向量相似度。向量均已 **L2 归一化** 后：
- **内积（Inner Product）等于余弦相似度**，因此 FAISS 使用 ``IndexFlatIP`` 即可等价于余弦检索。

联合检索 ``get_joint_feature`` 将图文向量在嵌入空间做凸组合（权重 ``alpha``），
再归一化，得到“以图为主、以文为辅”（或反之）的单一查询向量，实现简单的交互式检索融合。

安全加载说明（CVE-2025-32434）
------------------------------
``torch.load`` 在 PyTorch 2.6 前的 ``weights_only=True`` 路径仍存在风险；新版 ``transformers``
会在加载 pickle 权重（``pytorch_model.bin``）前强制检查 PyTorch 版本。

本项目默认 **优先使用 ``safetensors`` 权重**（``use_safetensors=True``），尽量避免触发 ``torch.load``；
若镜像/缓存缺失 safetensors，仅在 **torch>=2.6** 时回退到 ``pytorch_model.bin``。
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any, Callable, List, Sequence, TypeVar

import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

T = TypeVar("T")

# 强制使用国内 Hugging Face 镜像（需在首次触发 HF 下载前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 增大 hub 下载超时，降低大模型/大文件拉取时的超时概率（单位：秒）
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

logger = logging.getLogger(__name__)

# 与模型卡片一致的默认检查点（必须支持中文）
MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"


def _torch_ge_26() -> bool:
    """解析 ``torch.__version__``（兼容 ``2.6.0+cu124``）并判断是否满足 transformers 的安全加载门槛。"""
    ver = torch.__version__.split("+", 1)[0]
    parts: list[int] = []
    for token in ver.split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    while len(parts) < 3:
        parts.append(0)
    major, minor, patch = parts[0], parts[1], parts[2]
    return (major, minor, patch) >= (2, 6, 0)


def _extract_clip_feature_tensor(outputs: Any, *, modality: str) -> torch.Tensor:
    """
    将 ChineseCLIP / CLIP 系列模型的输出规整为形状 ``(batch, dim)`` 的 ``torch.Tensor``。

    说明：不同版本 ``transformers`` 中，``get_image_features`` / ``get_text_features``
    可能直接返回 Tensor，也可能返回 ``BaseModelOutputWithPooling`` 等包装对象。
    这里按常用字段优先级抽取，确保后续 ``_l2_normalize`` 始终接收 Tensor。
    """
    if isinstance(outputs, torch.Tensor):
        return outputs

    modality = modality.lower().strip()
    if modality not in {"image", "text"}:
        raise ValueError(f"modality 必须是 image/text，收到：{modality!r}")

    # 1) 最直接：Chinese CLIP 对齐空间的嵌入名称（若存在）
    if modality == "image":
        img = getattr(outputs, "image_embeds", None)
        if isinstance(img, torch.Tensor):
            return img
    else:
        txt = getattr(outputs, "text_embeds", None)
        if isinstance(txt, torch.Tensor):
            return txt

    # 2) ViT / BERT 类输出的 pooling 向量
    pooler = getattr(outputs, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        return pooler

    # 3) 序列输出：取 CLS（若存在）
    lhs = getattr(outputs, "last_hidden_state", None)
    if isinstance(lhs, torch.Tensor) and lhs.ndim == 3:
        return lhs[:, 0, :]

    # 4) 兼容 tuple/list（例如部分版本返回 (last_hidden_state, pooler_output)）
    if isinstance(outputs, (tuple, list)) and outputs:
        return _extract_clip_feature_tensor(outputs[0], modality=modality)

    raise TypeError(f"无法从模型输出中提取 {modality} 特征向量，类型={type(outputs)!r}")


class FeatureExtractor:
    """
    封装 Chinese CLIP 的前向与归一化逻辑，供建库与在线检索复用。
    """

    def __init__(self, model_id: str = MODEL_ID, device: torch.device | None = None) -> None:
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            logger.warning("未检测到 CUDA，将使用 CPU 推理（速度较慢）。建议在 GPU 环境运行。")

        if not _torch_ge_26():
            logger.warning(
                "当前 PyTorch=%s（<2.6）。建议升级到 torch>=2.6；"
                "否则一旦 safetensors 权重不可用，transformers 可能因 CVE-2025-32434 防护拒绝加载 pickle 权重。",
                torch.__version__,
            )

        # 模型下载/加载在网络不稳定环境下容易超时；这里做有限次重试 + 指数退避
        self.processor = self._from_pretrained_with_retry(
            lambda: ChineseCLIPProcessor.from_pretrained(self.model_id),
            what="processor",
        )
        self.model = self._from_pretrained_with_retry(self._load_chinese_clip_model, what="model")
        self.model.to(self.device)
        self.model.eval()
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache_max_size = 256

    def _load_chinese_clip_model(self) -> ChineseCLIPModel:
        """
        优先加载 ``*.safetensors``，绕开 ``torch.load`` 的安全版本门槛；必要时回退 bin。
        """
        try:
            return ChineseCLIPModel.from_pretrained(self.model_id, use_safetensors=True)
        except Exception as exc:  # noqa: BLE001
            if _torch_ge_26():
                logger.warning(
                    "ChineseCLIPModel：safetensors 加载失败，回退 pytorch_model.bin（torch=%s）：%s",
                    torch.__version__,
                    exc,
                )
                return ChineseCLIPModel.from_pretrained(self.model_id, use_safetensors=False)

            raise RuntimeError(
                "ChineseCLIPModel 加载失败：当前 PyTorch<2.6 时不允许走 pickle 权重加载路径（CVE-2025-32434）。\n"
                "可选解决方案：\n"
                "1) 升级 PyTorch 到 2.6+；或\n"
                "2) 确保缓存/镜像中存在 ``model.safetensors``，并能成功完成 ``use_safetensors=True`` 加载。\n"
                f"原始错误：{exc}"
            ) from exc

    def _from_pretrained_with_retry(self, loader: Callable[[], T], what: str, max_retries: int = 6) -> T:
        """
        对加载逻辑包一层重试：适配 HF 下载抖动；模型权重路径额外遵循 safetensors 优先策略。
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return loader()
            except Exception as exc:  # noqa: BLE001 - 需要兜住所有网络/解析/IO 异常
                last_exc = exc
                sleep_s = min(60.0, 2.0 ** (attempt - 1))
                logger.warning(
                    "加载 %s 失败（%d/%d）：%s；%.1fs 后重试…",
                    what,
                    attempt,
                    max_retries,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    @torch.inference_mode()
    def _l2_normalize(self, features: torch.Tensor) -> torch.Tensor:
        """对最后一维做 L2 归一化，避免零向量除零。"""
        denom = features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return features / denom

    @torch.inference_mode()
    def _get_text_feature_tensor(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        raw = self.model.get_text_features(**inputs)
        return self._l2_normalize(_extract_clip_feature_tensor(raw, modality="text"))

    @torch.inference_mode()
    def _get_image_feature_tensor(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        raw = self.model.get_image_features(**inputs)
        return self._l2_normalize(_extract_clip_feature_tensor(raw, modality="image"))

    @torch.inference_mode()
    def get_text_feature(self, text: str) -> np.ndarray:
        """
        单条文本 -> L2 归一化向量 (1, D)，float32 numpy。
        """
        key = text.strip()
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached.copy()

        feats = self._get_text_feature_tensor(key).detach().cpu().numpy().astype(np.float32)
        self._text_cache[key] = feats
        self._text_cache.move_to_end(key)
        if len(self._text_cache) > self._text_cache_max_size:
            self._text_cache.popitem(last=False)
        return feats.copy()

    @torch.inference_mode()
    def get_image_feature(self, image: Image.Image) -> np.ndarray:
        """
        单张 PIL 图像 -> L2 归一化向量 (1, D)。
        """
        feats = self._get_image_feature_tensor(image)
        return feats.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def get_joint_feature(
        self,
        image: Image.Image,
        text: str,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        联合查询向量：alpha * image + (1-alpha) * text，再 L2 归一化。

        alpha 接近 1 更侧重图像，接近 0 更侧重文本。
        """
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))

        img_t = self._get_image_feature_tensor(image)
        txt_np = self.get_text_feature(text)
        txt_t = torch.from_numpy(txt_np).to(self.device)
        fused = alpha * img_t + (1.0 - alpha) * txt_t
        fused = self._l2_normalize(fused)
        return fused.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_images_batch(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        批量图像编码（内部按 batch_size 切分），返回 (N, D) 的 float32 矩阵，已逐行 L2 归一化。
        用于大规模建库，避免一次性装入过多张图导致显存/内存峰值过高。
        """
        out_list: List[np.ndarray] = []
        n = len(images)
        for start in range(0, n, batch_size):
            chunk = list(images[start : start + batch_size])
            # 图像批次无需 text 侧的 padding；由 Processor 负责 resize / normalize / stack
            inputs = self.processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            raw = self.model.get_image_features(**inputs)
            feats = self._l2_normalize(_extract_clip_feature_tensor(raw, modality="image"))
            out_list.append(feats.detach().cpu().numpy().astype(np.float32))
        if not out_list:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(out_list)
