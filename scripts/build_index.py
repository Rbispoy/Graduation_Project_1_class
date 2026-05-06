"""
FAISS 向量索引构建脚本
======================

设计要点
--------
1. **批处理**：按 ``batch_size``（默认 64）读取图片并调用 ``FeatureExtractor.encode_images_batch``，
   控制 GPU 显存与主机内存峰值，适配 4 万级规模。
2. **扁平内积索引**：``faiss.IndexFlatIP``。向量已 L2 归一化时，内积等价于余弦相似度。
3. **顺序一致性**：将向量按文件名排序后依次 ``add`` 入索引，并写入 ``dataset/index_ids.json``，
   与 FAISS 行号一一对应，检索时才能把 ``search`` 返回的下标映射回商品 ``id``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 必须在 import core 之前把项目根目录加入 sys.path（从 scripts/ 直接运行时会找不到包）
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_sys_path()

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from core.feature_extractor import FeatureExtractor


def _list_image_paths(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.name)
    return paths


def main() -> None:
    root = _project_root()
    images_dir = root / "dataset" / "images"
    index_path = root / "dataset" / "ecommerce.index"
    id_map_path = root / "dataset" / "index_ids.json"

    batch_size = 64

    paths = _list_image_paths(images_dir)
    if not paths:
        raise SystemExit(f"未找到图像：{images_dir}。请先运行 scripts/download_data.py")

    print("加载 Chinese CLIP 并批量提特征（耗时取决于 GPU / CPU）…")
    extractor = FeatureExtractor()

    # 投影维度：优先读取线性层输出特征数（与 HF 实现一致），兼容旧版 transformers
    proj = getattr(extractor.model, "text_projection", None)
    if proj is not None:
        dim = int(proj.out_features)
    else:
        dim = int(getattr(extractor.model.config, "projection_dim", 512))

    index = faiss.IndexFlatIP(dim)
    id_order: list[str] = []

    for start in tqdm(range(0, len(paths), batch_size), desc="建库 batch"):
        batch_paths = paths[start : start + batch_size]
        pil_list: list[Image.Image] = []
        for p in batch_paths:
            with Image.open(p) as im:
                pil_list.append(im.convert("RGB"))
        feats = extractor.encode_images_batch(pil_list, batch_size=len(pil_list))
        # FAISS 要求 float32 连续内存
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        index.add(feats)
        id_order.extend(p.stem for p in batch_paths)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with id_map_path.open("w", encoding="utf-8") as f:
        json.dump(id_order, f, ensure_ascii=False, indent=2)

    print(f"索引已写入：{index_path}")
    print(f"ID 顺序表：{id_map_path}（长度 {len(id_order)}，维度 {dim}）")


if __name__ == "__main__":
    main()
