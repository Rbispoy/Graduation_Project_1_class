"""
电商图像数据集下载脚本
======================

数据来源：Hugging Face ``ashraq/fashion-product-images-small``（约 4.4 万条，``split="train"`` 全量）。

网络策略
--------
- 通过 ``HF_ENDPOINT`` 指向国内镜像，减轻直连 huggingface.co 的不稳定。
- 清空 ``HTTP_PROXY`` / ``HTTPS_PROXY``，避免失效代理劫持会话。
- 对需 HTTP 拉取的图像使用 ``requests.Session()``（连接复用、统一超时与重试策略）。

断点续传
--------
若 ``./dataset/images/{id}.jpg`` 已存在则跳过写入，仅更新元数据；单条失败不影响整体跑完。
"""

from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# 须在导入 huggingface_hub / datasets 之前生效，确保镜像端点被正确识别
# （遵循「先完成基础 import，再立刻设置环境变量」的写法）
# ---------------------------------------------------------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

from datasets import load_dataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    """保证从 ``scripts/`` 直接运行时也能 ``import core``（PyCharm 调试常见场景）。"""
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _build_session() -> requests.Session:
    """
    集中管理 HTTP 连接：Session 复用 TCP、统一头与重试，降低间歇性失败概率。
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Multimodal-Joint-Retrieval/1.0 (datasets-downloader)",
            "Accept": "image/*,*/*;q=0.8",
        }
    )
    return session


def _pick(row: dict, *keys: str, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _normalize_id(raw) -> str:
    if raw is None:
        raise ValueError("缺少商品 ID")
    return str(raw).strip()


def _image_to_rgb_pil(raw, session: requests.Session) -> Image.Image:
    """
    将 Hugging Face ``datasets`` 中常见的多种图像表示转为 RGB 的 ``PIL.Image``。

    - 已是 ``PIL.Image`` / ``numpy``：本地解码。
    - ``{"bytes": ..., "path": ...}``：优先内存字节；``path`` 为 URL 时用 ``session`` 拉取。
    """
    if raw is None:
        raise ValueError("图像字段为空")

    if isinstance(raw, Image.Image):
        im = raw
    elif isinstance(raw, np.ndarray):
        im = Image.fromarray(raw)
    elif hasattr(raw, "__array__"):
        im = Image.fromarray(np.asarray(raw))
    elif isinstance(raw, (bytes, bytearray)):
        im = Image.open(BytesIO(raw))
    elif isinstance(raw, dict):
        blob = raw.get("bytes")
        path = raw.get("path")
        if blob:
            im = Image.open(BytesIO(blob))
        elif path:
            p = str(path)
            if p.startswith(("http://", "https://")):
                resp = session.get(p, timeout=(10, 120))
                resp.raise_for_status()
                im = Image.open(BytesIO(resp.content))
            else:
                im = Image.open(p)
        else:
            raise ValueError("图像字典缺少可用的 bytes 或 path")
    else:
        raise TypeError(f"无法识别的图像类型: {type(raw)}")

    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def main() -> None:
    _ensure_sys_path()

    root = _project_root()
    images_dir = root / "dataset" / "images"
    metadata_path = root / "dataset" / "metadata.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = _build_session()

    try:
        print(
            "正在从 Hugging Face 加载数据集（全量 train；镜像: "
            f"{os.environ.get('HF_ENDPOINT', '默认端点')}）…"
        )
        ds = load_dataset("ashraq/fashion-product-images-small", split="train")

        try:
            total_rows = len(ds)
        except TypeError:
            total_rows = None

        metadata_by_id: dict[str, dict] = {}
        written_this_run = 0
        skipped_existing = 0
        failed = 0

        loop_start = time.perf_counter()
        last_log_time = loop_start
        last_log_idx = 0

        with tqdm(
            total=total_rows,
            desc="下载图像",
            unit="img",
            dynamic_ncols=True,
            mininterval=0.3,
        ) as pbar:
            for idx, row in enumerate(ds):
                raw_id = _pick(row, "id", "ID", default=idx)
                try:
                    item_id = _normalize_id(raw_id)
                except ValueError as e:
                    tqdm.write(f"[跳过] 第 {idx} 行 ID 无效: {e}")
                    failed += 1
                    pbar.update(1)
                    elapsed = time.perf_counter() - loop_start
                    speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                    pbar.set_postfix(
                        已写入=written_this_run,
                        跳过已存在=skipped_existing,
                        失败=failed,
                        速度=f"{speed:.2f} img/s",
                        refresh=False,
                    )
                    continue

                rel_name = f"{item_id}.jpg"
                out_path = images_dir / rel_name

                meta = {
                    "id": item_id,
                    "productDisplayName": _pick(row, "productDisplayName", "product_display_name", default="") or "",
                    "masterCategory": _pick(row, "masterCategory", "master_category", default="") or "",
                    "subCategory": _pick(row, "subCategory", "sub_category", default="") or "",
                    "gender": _pick(row, "gender", default="") or "",
                    "articleType": _pick(row, "articleType", "article_type", default="") or "",
                    "baseColour": _pick(row, "baseColour", "base_colour", default="") or "",
                    "image_file": rel_name,
                }
                metadata_by_id[item_id] = meta

                if out_path.exists():
                    skipped_existing += 1
                else:
                    try:
                        raw_img = _pick(row, "image")
                        if raw_img is None:
                            raise ValueError("无 image 字段")

                        pil_img = _image_to_rgb_pil(raw_img, session)
                        pil_img.save(out_path, format="JPEG", quality=95, optimize=True)
                        written_this_run += 1
                    except requests.RequestException as exc:
                        tqdm.write(f"[网络错误] idx={idx} id={item_id}: {exc}")
                        failed += 1
                    except Exception as exc:
                        tqdm.write(f"[错误] idx={idx} id={item_id}: {type(exc).__name__}: {exc}")
                        failed += 1

                pbar.update(1)
                elapsed = time.perf_counter() - loop_start
                speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix(
                    已写入=written_this_run,
                    跳过已存在=skipped_existing,
                    失败=failed,
                    速度=f"{speed:.2f} img/s",
                    refresh=False,
                )

                now = time.perf_counter()
                if now - last_log_time >= 2.0:
                    delta = now - last_log_time
                    done_window = (idx + 1) - last_log_idx
                    inst_speed = done_window / delta if delta > 0 else 0.0
                    tqdm.write(
                        f"[进度] {idx + 1}/{total_rows or '?'} 行 | "
                        f"本运行新写入 {written_this_run} | 跳过 {skipped_existing} | 失败 {failed} | "
                        f"近期 {inst_speed:.2f} img/s | 累计平均 {speed:.2f} img/s"
                    )
                    last_log_time = now
                    last_log_idx = idx + 1

        metadata_list = sorted(metadata_by_id.values(), key=lambda m: m["id"])
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        total_elapsed = time.perf_counter() - loop_start
        print(
            f"完成：图像目录 {images_dir}，元数据 {metadata_path}，"
            f"元数据条数 {len(metadata_list)}；"
            f"本运行新写入 {written_this_run}，断点跳过 {skipped_existing}，失败 {failed}；"
            f"总耗时 {total_elapsed:.1f}s。"
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
