"""
从 Kaggle「Fashion Product Images Dataset」完整版导入（答辩用高清主图）
================================================================

与 HF 上的 ``ashraq/fashion-product-images-small`` 同源体系，但 Kaggle **完整版**里一般是
**更大尺寸的电商主图**，观感明显好于 Small 里的压缩缩略图。

你需要做的
--------
1. 打开 Kaggle 数据集页，登录并接受条款后下载（约数 GB，视网速需若干小时）：
   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
2. 解压到本地任意目录，例如 ``E:\\data\\fashion-product-images-dataset``，
   确保解压后能看到 ``styles.csv`` 和 ``images`` 文件夹（或含 ``images.csv`` 的常见布局）。

3. 在本项目根目录执行（示例：抽 6000 张、宽高至少 500）：

   python scripts/import_kaggle_fashion_full.py --source "E:\\data\\fashion-product-images-dataset" --clean --max-rows 6000 --min-width 500 --min-height 500

4. 再运行 ``python scripts/build_index.py`` 与 ``python main.py``。

说明
----
- ``--clean``：先清空 ``dataset/images/*.jpg``，避免与旧 Small 数据混用。
- 若解压目录结构略有不同，可用 ``--styles-csv`` / ``--images-dir`` 手动指定。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _find_default_paths(source: Path) -> tuple[Path, Path]:
    styles = source / "styles.csv"
    images_dir = source / "images"
    if not styles.is_file():
        raise SystemExit(f"未找到 styles.csv：{styles}\n请检查 --source 是否为解压后的数据集根目录。")
    if not images_dir.is_dir():
        raise SystemExit(f"未找到 images 目录：{images_dir}")
    return styles, images_dir


def _load_image_map(images_csv: Path) -> dict[str, str]:
    """若存在 images.csv，建立 id -> 相对文件名（在 images 目录下）。"""
    out: dict[str, str] = {}
    with images_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            fn = (row.get("filename") or row.get("image") or row.get("file_name") or "").strip()
            if fn:
                out[rid] = fn
    return out


def _resolve_image_path(
    images_dir: Path,
    item_id: str,
    id_to_file: dict[str, str],
) -> Path | None:
    if item_id in id_to_file:
        rel = id_to_file[item_id].replace("\\", "/").lstrip("/")
        p = images_dir / rel
        if p.is_file():
            return p
        p2 = images_dir / Path(rel).name
        if p2.is_file():
            return p2
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = images_dir / f"{item_id}{ext}"
        if cand.is_file():
            return cand
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 Kaggle Fashion Product Images 完整版导入到 dataset/")
    p.add_argument("--source", type=Path, required=True, help="Kaggle 解压后的数据集根目录")
    p.add_argument("--styles-csv", type=Path, default=None, help="可选：styles.csv 路径")
    p.add_argument("--images-dir", type=Path, default=None, help="可选：images 文件夹路径")
    p.add_argument("--images-csv", type=Path, default=None, help="可选：images.csv 路径（默认 source/images.csv）")
    p.add_argument("--max-rows", type=int, default=None, help="最多导入多少条（shuffle 后截取）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-width", type=int, default=0)
    p.add_argument("--min-height", type=int, default=0)
    p.add_argument("--clean", action="store_true", help="导入前删除 dataset/images 下所有 .jpg")
    return p.parse_args()


def main() -> None:
    _ensure_sys_path()
    args = _parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"--source 不是目录：{source}")

    if args.styles_csv and args.images_dir:
        styles_path = args.styles_csv.resolve()
        images_dir = args.images_dir.resolve()
    else:
        styles_path, images_dir = _find_default_paths(source)

    images_csv = args.images_csv
    if images_csv is None:
        cand = source / "images.csv"
        images_csv = cand if cand.is_file() else None
    id_to_file: dict[str, str] = {}
    if images_csv and Path(images_csv).is_file():
        id_to_file = _load_image_map(Path(images_csv))
        print(f"已读取 images.csv，映射条数 {len(id_to_file)}")

    root = _project_root()
    out_images = root / "dataset" / "images"
    meta_path = root / "dataset" / "metadata.json"
    out_images.mkdir(parents=True, exist_ok=True)

    if args.clean:
        n = 0
        for p in out_images.glob("*.jpg"):
            p.unlink(missing_ok=True)
            n += 1
        print(f"[clean] 已删除 {n} 个 .jpg")

    rows: list[dict[str, str]] = []
    with styles_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            rows.append({k: (v if v is not None else "") for k, v in row.items()})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_rows is not None and args.max_rows > 0:
        rows = rows[: min(args.max_rows, len(rows))]
    print(f"待处理样式条数：{len(rows)}（styles.csv: {styles_path}）")

    metadata: list[dict] = []
    copied = 0
    skipped_res = 0
    missing = 0

    for row in tqdm(rows, desc="导入图片", unit="row"):
        rid = str(row.get("id", "")).strip()
        img_path = _resolve_image_path(images_dir, rid, id_to_file)
        if img_path is None:
            missing += 1
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if (args.min_width and w < args.min_width) or (args.min_height and h < args.min_height):
                    skipped_res += 1
                    continue
                dest = out_images / f"{rid}.jpg"
                im.save(dest, format="JPEG", quality=95, optimize=True)
        except OSError:
            missing += 1
            continue

        copied += 1

        metadata.append(
            {
                "id": rid,
                "productDisplayName": (row.get("productDisplayName") or "").strip(),
                "masterCategory": (row.get("masterCategory") or "").strip(),
                "subCategory": (row.get("subCategory") or "").strip(),
                "gender": (row.get("gender") or "").strip(),
                "articleType": (row.get("articleType") or "").strip(),
                "baseColour": (row.get("baseColour") or "").strip(),
                "image_file": f"{rid}.jpg",
            }
        )

    metadata.sort(key=lambda m: m["id"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"完成：复制 {copied} 张到 {out_images}，metadata {len(metadata)} 条写入 {meta_path}；"
        f"分辨率过滤 {skipped_res}，找不到图 {missing}。"
    )
    print("下一步：python scripts/build_index.py")


if __name__ == "__main__":
    main()
