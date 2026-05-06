"""
FastAPI 多模态检索服务
=====================

静态资源
--------
- ``IMAGES_URL_PREFIX``（默认 ``/images``）：挂载到 ``dataset/images``；搜索接口返回的 ``image_url``
  与此前缀拼接，保证与 ``IMAGES_DIR`` 一致。

检索接口 ``POST /api/search``
-----------------------------
``multipart/form-data`` 字段：
- ``query``：可选，文本查询。
- ``file``：可选，上传图像。
- ``alpha``：可选，默认 0.6；仅当 **图文同时提供** 且未开 ``auto_alpha`` 时用于 ``get_joint_feature``。
- ``auto_alpha``：可选，默认 false；为 true 且图文同时提供时，由服务端按文本长度与图像清晰度 **自适应** 计算 α（见 ``core/adaptive_alpha.py``）。

分支逻辑
--------
1. 仅文本：文本特征检索（中文）。
2. 仅图像：图像特征检索。
3. 图文兼有：联合向量检索，融合两种模态。

创新点扩展（C，可选）
--------------------
当前线上推理为 **Chinese CLIP 零样本检索**；若需更强领域适配，可在本仓库外对视觉或文本塔做 **LoRA / Prompt Tuning**
（需独立训练脚本与标注对，答辩中可作为「未来工作」简述）。
"""

from __future__ import annotations

import os
# 1. 强制走国内镜像站，解决连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. 绕过 PyTorch 2.5.1 的安全检查，允许加载模型权重
os.environ["TORCH_FORCE_SAFE_LOAD"] = "0"

import concurrent.futures
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import faiss
import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core.adaptive_alpha import compute_adaptive_alpha
from core.feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 关闭第三方依赖的高频网络探测日志，仅保留告警/错误。
# 例如 httpx 对 HF 镜像发起的 HEAD/GET 探测通常是正常行为，但会刷屏。
for noisy_name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(noisy_name).setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
# 与 StaticFiles 挂载路径一致；API 返回的 image_url 必须为「挂载前缀 + 文件名」
IMAGES_URL_PREFIX = os.getenv("IMAGES_URL_PREFIX", "images").strip("/") or "images"
IMAGES_ROUTE = f"/{IMAGES_URL_PREFIX}"
METADATA_PATH = DATASET_DIR / "metadata.json"
INDEX_PATH = DATASET_DIR / "ecommerce.index"
ID_MAP_PATH = DATASET_DIR / "index_ids.json"
FRONTEND_INDEX = ROOT / "frontend" / "index.html"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="Multimodal CLIP + FAISS Search",
    version="1.0.0",
    lifespan=_lifespan,
)

# 本地单页调试时浏览器直连 API 可能需要 CORS；生产环境请收紧 allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_extractor: Optional[FeatureExtractor] = None
_index: Optional[faiss.Index] = None
_id_order: list[str] = []
_meta_by_id: dict[str, dict[str, Any]] = {}
_flat_vectors: Optional[np.ndarray] = None
_TOP_K = 20


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_index_and_meta() -> tuple[faiss.Index, list[str], dict[str, dict[str, Any]]]:
    """读 FAISS + ID 映射 + 元数据（可与模型加载并行）。"""
    t0 = time.perf_counter()
    index = faiss.read_index(str(INDEX_PATH))
    with ID_MAP_PATH.open(encoding="utf-8") as f:
        id_order = json.load(f)
    if index.ntotal != len(id_order):
        logger.warning("索引向量数与 ID 映射长度不一致，请重新建库。")
    items = _load_json_list(METADATA_PATH)
    meta_by_id = {str(m.get("id")): m for m in items if m.get("id") is not None}
    logger.info("索引与元数据就绪：%d 条元数据，耗时 %.2fs", len(meta_by_id), time.perf_counter() - t0)
    return index, id_order, meta_by_id


def _load_extractor() -> FeatureExtractor:
    t0 = time.perf_counter()
    ex = FeatureExtractor()
    logger.info("Chinese CLIP 就绪，耗时 %.2fs", time.perf_counter() - t0)
    return ex



def load_resources() -> None:
    """启动时并行加载模型与索引，缩短 wall-clock；瓶颈通常仍在 CLIP 权重 I/O + 初始化。"""
    global _extractor, _index, _id_order, _meta_by_id, _flat_vectors

    if not INDEX_PATH.exists() or not ID_MAP_PATH.exists():
        raise RuntimeError(
            f"缺少索引文件：{INDEX_PATH} 或 {ID_MAP_PATH}。请先运行 scripts/build_index.py"
        )

    wall = time.perf_counter()
    logger.info("并行加载：Chinese CLIP + FAISS/元数据 …")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_clip = pool.submit(_load_extractor)
        fut_index = pool.submit(_load_index_and_meta)
        _extractor = fut_clip.result()
        _index, _id_order, _meta_by_id = fut_index.result()

    _flat_vectors = None  # 暴力对比按需懒加载
    logger.info(
        "启动资源加载完成，总 wall 时间 %.2fs（主要耗时在 Chinese CLIP 权重加载；"
        "ViT-B 级首次约 10～20s 常见，第二次启动若走磁盘缓存会略短）",
        time.perf_counter() - wall,
    )


def _public_image_url(image_file: str, *, item_id: str) -> str:
    """
    生成浏览器可直接访问的图片 URL，与 ``app.mount(IMAGES_URL_PREFIX, StaticFiles(IMAGES_DIR))`` 对齐。

    - 使用文件名（basename），忽略元数据中可能带入的子路径或重复前缀。
    - 对文件名做 URL 编码，兼容空格等特殊字符。
    """
    raw = (image_file or "").strip().replace("\\", "/").lstrip("/")
    if raw.lower().startswith("images/"):
        raw = raw[7:]
    name = Path(raw).name if raw else ""
    if not name:
        name = f"{item_id}.jpg"
    safe = quote(name, safe="")
    return f"./{IMAGES_URL_PREFIX}/{safe}"


# 必须先确保目录存在再挂载，否则启动时若文件夹尚未创建会导致整条 /images 路由丢失、前端永久裂图
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount(IMAGES_ROUTE, StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/")
def serve_frontend() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html 不存在")
    return FileResponse(str(FRONTEND_INDEX))


def _parse_optional_str(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    return s if s else None


async def _read_upload_image(file: UploadFile | None) -> Optional[Image.Image]:
    if file is None:
        return None
    # 未上传文件时 filename 可能为空
    if not file.filename:
        return None
    data = await file.read()
    if not data:
        return None
    return Image.open(BytesIO(data)).convert("RGB")


def _search_vector(vec: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    assert _index is not None
    q = np.ascontiguousarray(vec.astype(np.float32), dtype=np.float32)
    t0 = time.perf_counter()
    scores, indices = _index.search(q, _TOP_K)
    faiss_ms = (time.perf_counter() - t0) * 1000.0

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        item_id = _id_order[idx]
        meta = dict(_meta_by_id.get(item_id, {"id": item_id}))
        image_file = meta.get("image_file") or f"{item_id}.jpg"
        results.append(
            {
                **meta,
                "score": float(score),
                "image_url": _public_image_url(str(image_file), item_id=str(item_id)),
            }
        )
    return results, faiss_ms


def _ensure_flat_vectors_for_benchmark() -> None:
    """按需构建与 IndexFlatIP 对齐的向量矩阵，供 NumPy 暴力基线计时。"""
    global _flat_vectors  # noqa: PLW0603
    if _flat_vectors is not None:
        return
    if _index is None or not isinstance(_index, faiss.IndexFlatIP):
        return
    try:
        xb_ptr = _index.get_xb()
        _flat_vectors = faiss.rev_swig_ptr(xb_ptr, _index.ntotal * _index.d).reshape(_index.ntotal, _index.d)
        logger.info("已按需加载扁平向量矩阵用于 benchmark，shape=%s", _flat_vectors.shape)
    except Exception as exc:  # noqa: BLE001
        logger.warning("构建暴力检索基线矩阵失败，benchmark 将跳过对比：%s", exc)


def _brute_force_topk_ms(vec: np.ndarray) -> Optional[float]:
    """NumPy 暴力内积基线耗时，仅用于答辩/调优对比。"""
    _ensure_flat_vectors_for_benchmark()
    if _flat_vectors is None:
        return None
    q = np.ascontiguousarray(vec.astype(np.float32), dtype=np.float32)
    t0 = time.perf_counter()
    sims = _flat_vectors @ q[0]
    _ = np.argpartition(-sims, _TOP_K - 1)[:_TOP_K]
    return (time.perf_counter() - t0) * 1000.0


@app.post("/api/search")
async def api_search(
    query: Optional[str] = Form(None),
    file: UploadFile | None = File(None),
    alpha: float = Form(0.6),
    auto_alpha: bool = Form(False),
    benchmark: bool = Form(False),
) -> dict[str, Any]:
    """
    多模态检索：根据表单字段自动选择 文本 / 图像 / 联合 查询向量。
    """
    text = _parse_optional_str(query)
    image = await _read_upload_image(file)

    if text is None and image is None:
        raise HTTPException(status_code=400, detail="请至少提供文本或图片之一")

    assert _extractor is not None

    joint = text is not None and image is not None
    alpha_used = float(alpha)
    alpha_auto = False
    if joint and auto_alpha:
        alpha_used = compute_adaptive_alpha(text, image)
        alpha_auto = True

    t_start = time.perf_counter()
    if joint:
        vec = _extractor.get_joint_feature(image, text, alpha=alpha_used)
    elif text is not None:
        vec = _extractor.get_text_feature(text)
    else:
        assert image is not None
        vec = _extractor.get_image_feature(image)
    embed_ms = (time.perf_counter() - t_start) * 1000.0

    hits, faiss_ms = _search_vector(vec)
    total_ms = (time.perf_counter() - t_start) * 1000.0
    perf: dict[str, Any] = {
        "embed_ms": round(embed_ms, 2),
        "faiss_ms": round(faiss_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    if joint:
        perf["alpha_used"] = round(alpha_used, 4)
        perf["alpha_auto"] = alpha_auto
        if not alpha_auto:
            perf["alpha_manual"] = round(float(alpha), 4)
    if benchmark:
        brute_ms = _brute_force_topk_ms(vec)
        if brute_ms is not None:
            perf["bruteforce_ms"] = round(brute_ms, 2)
            perf["speedup_vs_bruteforce"] = round(brute_ms / max(faiss_ms, 1e-6), 2)
    return {"results": hits, "top_k": len(hits), "perf": perf}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
