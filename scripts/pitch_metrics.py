"""
答辩用离线指标（创新点 A：可量化评测）
====================================

在 **已建好索引** 的前提下运行（需 ``dataset/ecommerce.index``、``index_ids.json``、``metadata.json``）。

输出两类数字，方便直接写进 PPT：
1. **文本自检索 Recall**：用商品名 ``productDisplayName`` 作为查询，看真值 id 是否落在 Top-K（检验库与编码是否对齐）。
2. **延迟统计**：随机抽若干条文本，统计编码 + FAISS 的耗时均值（毫秒）。

用法::

    python scripts/pitch_metrics.py --sample 300 --top-k 20 --bench 50
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

import argparse
import json
import random
import sys
import time
from pathlib import Path

def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    r = _root()
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))


_ensure_sys_path()

import faiss
import numpy as np

from core.feature_extractor import FeatureExtractor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300, help="参与 Recall 评测的样本数")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--bench", type=int, default=50, help="延迟压测条数（0 表示跳过）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = _root()
    meta_path = root / "dataset" / "metadata.json"
    index_path = root / "dataset" / "ecommerce.index"
    id_path = root / "dataset" / "index_ids.json"

    if not meta_path.is_file() or not index_path.is_file() or not id_path.is_file():
        raise SystemExit("缺少 metadata.json / ecommerce.index / index_ids.json，请先 build_index。")

    with meta_path.open(encoding="utf-8") as f:
        meta_list = json.load(f)
    with id_path.open(encoding="utf-8") as f:
        id_order = json.load(f)

    rows = [m for m in meta_list if m.get("id") is not None and (m.get("productDisplayName") or "").strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: min(args.sample, len(rows))]
    if not rows:
        raise SystemExit("metadata 中没有可用的 productDisplayName。")

    print("加载模型与索引 …")
    extractor = FeatureExtractor()
    index = faiss.read_index(str(index_path))

    hit1 = 0
    hitk = 0
    for m in rows:
        qid = str(m["id"])
        text = str(m["productDisplayName"]).strip()
        vec = extractor.get_text_feature(text)
        q = np.ascontiguousarray(vec.astype(np.float32), dtype=np.float32)
        scores, indices = index.search(q, args.top_k)
        ids = [id_order[i] for i in indices[0].tolist() if 0 <= i < len(id_order)]
        if ids and ids[0] == qid:
            hit1 += 1
        if qid in ids:
            hitk += 1

    n = len(rows)
    print("--- 文本自检索（商品名 -> 自身 id）---")
    print(f"样本数: {n}  |  Top-K = {args.top_k}")
    print(f"Recall@1  = {hit1 / n:.4f}")
    print(f"Recall@{args.top_k} = {hitk / n:.4f}")

    if args.bench <= 0:
        return

    texts = [str(m["productDisplayName"]).strip() for m in rows[: min(args.bench, len(rows))]]
    embed_ms: list[float] = []
    faiss_ms: list[float] = []
    for t in texts:
        t0 = time.perf_counter()
        vec = extractor.get_text_feature(t)
        t1 = time.perf_counter()
        q = np.ascontiguousarray(vec.astype(np.float32), dtype=np.float32)
        t2 = time.perf_counter()
        index.search(q, args.top_k)
        t3 = time.perf_counter()
        embed_ms.append((t1 - t0) * 1000.0)
        faiss_ms.append((t3 - t2) * 1000.0)

    print("--- 延迟压测（文本编码 + FAISS）---")
    print(f"条数: {len(texts)}")
    print(f"编码均值: {sum(embed_ms) / len(embed_ms):.2f} ms")
    print(f"FAISS 均值: {sum(faiss_ms) / len(faiss_ms):.2f} ms")


if __name__ == "__main__":
    main()
