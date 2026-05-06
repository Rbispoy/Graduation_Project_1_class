"""
电商服饰图像数据集下载脚本
==========================

**仅用于「服饰商品图文检索 / 服饰搜索引擎」类课题**：从 Hugging Face 拉取带 ``image`` 与商品字段的服饰数据集，
写入 ``dataset/images`` 与 ``dataset/metadata.json``。

默认：``ashraq/fashion-product-images-small``（Fashion Product Images Small）。

觉得缩略图偏糊时，可在**仍是服饰数据**的前提下：
- 用 ``--min-width`` / ``--min-height`` 过滤掉过小的图；
- 或换另一套 HF 服饰集（``--dataset mecha2019/fashion-product-images-small`` 等，字段需相近）。

示例
----
.. code-block:: text

   # 默认：服饰 ashraq 全量 train
   python scripts/download_data.py

   # 子集 + 清空旧图 + 只要偏大图（观感更好，仍是服饰）
   python scripts/download_data.py --clean --max-rows 5000 --min-width 400 --min-height 400

   # 另一套 HF 服饰数据
   python scripts/download_data.py --dataset mecha2019/fashion-product-images-small --clean --max-rows 5000

   # 答辩要更清晰主图：用 Kaggle「Fashion Product Images Dataset」完整版（非 Small），见
   # scripts/import_kaggle_fashion_full.py

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

import argparse
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 Hugging Face 下载服饰商品图并生成 metadata.json")
    p.add_argument(
        "--dataset",
        default="ashraq/fashion-product-images-small",
        help="HF 服饰数据集名，默认 ashraq/fashion-product-images-small",
    )
    p.add_argument("--split", default="train", help="数据划分名，默认 train")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="最多处理多少条（先 shuffle 再截取，便于答辩子集）",
    )
    p.add_argument("--seed", type=int, default=42, help="shuffle 随机种子")
    p.add_argument(
        "--min-width",
        type=int,
        default=0,
        help="仅保留宽不小于该值的样本（0 表示不限制；需解码图像后判断）",
    )
    p.add_argument(
        "--min-height",
        type=int,
        default=0,
        help="仅保留高不小于该值的样本（0 表示不限制）",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="开始前删除 dataset/images 下所有 .jpg（换库时建议开启）",
    )
    return p.parse_args()


def main() -> None:
    _ensure_sys_path()
    args = _parse_args()

    root = _project_root()
    images_dir = root / "dataset" / "images"
    metadata_path = root / "dataset" / "metadata.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        removed = 0
        for p in images_dir.glob("*.jpg"):
            p.unlink(missing_ok=True)
            removed += 1
        print(f"[clean] 已删除 {removed} 个 .jpg")

    session = _build_session()

    try:
        print(
            f"正在从 Hugging Face 加载数据集 {args.dataset!r} split={args.split!r}（镜像: "
            f"{os.environ.get('HF_ENDPOINT', '默认端点')}）…"
        )
        ds = load_dataset(args.dataset, split=args.split)
        if args.max_rows is not None and args.max_rows > 0:
            n = min(args.max_rows, len(ds))
            ds = ds.shuffle(seed=args.seed).select(range(n))
            print(f"已抽样子集：{n} 条（seed={args.seed}）")

        try:
            total_rows = len(ds)
        except TypeError:
            total_rows = None

        metadata_by_id: dict[str, dict] = {}
        written_this_run = 0
        skipped_existing = 0
        failed = 0
        skipped_small = 0

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
                        分辨率过滤=skipped_small,
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
                        raw_img = _pick(row, "image", "Image")
                        if raw_img is None:
                            raise ValueError("无 image 字段")

                        pil_img = _image_to_rgb_pil(raw_img, session)
                        w, h = pil_img.size
                        if (args.min_width and w < args.min_width) or (args.min_height and h < args.min_height):
                            skipped_small += 1
                            del metadata_by_id[item_id]
                            pbar.update(1)
                            elapsed = time.perf_counter() - loop_start
                            speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                            pbar.set_postfix(
                                已写入=written_this_run,
                                跳过已存在=skipped_existing,
                                分辨率过滤=skipped_small,
                                失败=failed,
                                速度=f"{speed:.2f} img/s",
                                refresh=False,
                            )
                            continue
                        pil_img.save(out_path, format="JPEG", quality=95, optimize=True)
                        written_this_run += 1
                    except requests.RequestException as exc:
                        tqdm.write(f"[网络错误] idx={idx} id={item_id}: {exc}")
                        metadata_by_id.pop(item_id, None)
                        failed += 1
                    except Exception as exc:
                        tqdm.write(f"[错误] idx={idx} id={item_id}: {type(exc).__name__}: {exc}")
                        metadata_by_id.pop(item_id, None)
                        failed += 1

                pbar.update(1)
                elapsed = time.perf_counter() - loop_start
                speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix(
                    已写入=written_this_run,
                    跳过已存在=skipped_existing,
                    分辨率过滤=skipped_small,
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
                        f"本运行新写入 {written_this_run} | 跳过 {skipped_existing} | "
                        f"分辨率过滤 {skipped_small} | 失败 {failed} | "
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
            f"本运行新写入 {written_this_run}，断点跳过 {skipped_existing}，"
            f"分辨率过滤 {skipped_small}，失败 {failed}；"
            f"总耗时 {total_elapsed:.1f}s。"
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
