# 电商图文联合检索（Chinese CLIP + FAISS）

基于 **OFA-Sys/chinese-clip-vit-base-patch16** 提取图文特征，**FAISS IndexFlatIP** 做相似检索；支持纯文本、纯图像查询，以及图文同时在嵌入空间内的加权融合（含自适应融合权重 α）。

---

## 环境要求

- **Python 3.10+**
- **操作系统**：Windows / Linux / macOS 均可  
- **GPU**：可选；无 CUDA 时自动用 CPU（首次加载模型较慢，检索仍可演示）
- 首次运行会从镜像下载模型权重，请保持网络可用。

---

## 1. 安装依赖

在项目根目录执行：

```bash
python -m venv .venv
```

**Windows（PowerShell）：**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

**Linux / macOS：**

```bash
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

若使用 **NVIDIA GPU**，请先到 [PyTorch 官网](https://pytorch.org/) 按本机 CUDA 版本安装对应的 `torch`，再执行 `pip install -r requirements.txt`（或先装 `torch` 再装其余依赖），以启用 GPU 推理。

---

## 2. 准备数据

将商品图与元数据写入 `dataset/images` 与 `dataset/metadata.json`。若本地还没有数据，可用脚本从 Hugging Face 拉取服饰类示例集（已默认使用国内 `hf-mirror` 端点）：

```bash
python scripts/download_data.py
```

子集示例（条数少、下载快，适合先跑通流程）：

```bash
python scripts/download_data.py --clean --max-rows 2000
```

更多参数见 `scripts/download_data.py` 文件顶部说明。

---

## 3. 构建向量索引

确保 `dataset/images` 下已有图片后执行：

```bash
python scripts/build_index.py
```

成功后会生成：

- `dataset/ecommerce.index` — FAISS 索引  
- `dataset/index_ids.json` — 索引行号与商品 `id` 的对应关系  

（脚本会从现有 `metadata.json` 与图片文件名对齐；详细逻辑见 `scripts/build_index.py`。）

---

## 4. 启动 Web 服务

在项目根目录：

```bash
python main.py
```

默认监听 **`http://0.0.0.0:8000`**。

浏览器访问：**[http://127.0.0.1:8000](http://127.0.0.1:8000)** —— 单页前端由 FastAPI 直接返回；检索接口为 `POST /api/search`。

若要指定端口（示例 8080）：

```bash
# 可临时改 main.py 末尾 uvicorn.run 的 port，或使用：
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## 5. 答辩 / 报告用离线指标（可选）

在已建好 `ecommerce.index`、`index_ids.json`、`metadata.json` 的前提下：

```bash
python scripts/pitch_metrics.py --sample 300 --top-k 20 --bench 50
```

会输出文本自检索 Recall 与编码、FAISS 延迟等统计，便于写入论文或 PPT。

---

## 目录概览

| 路径 | 说明 |
|------|------|
| `main.py` | FastAPI 服务：静态页、图片目录挂载、`/api/search` |
| `frontend/index.html` | 检索界面（Vue CDN） |
| `core/feature_extractor.py` | Chinese CLIP 特征与联合向量 |
| `core/adaptive_alpha.py` | 图文联合时自适应 α |
| `dataset/` | 图片、元数据、索引（构建后产生） |
| `scripts/` | 下载数据、建索引、离线指标 |

---

## 常见问题

1. **连接 Hugging Face 超时**  
   项目已设置 `HF_ENDPOINT=https://hf-mirror.com`；若仍失败，可检查本机代理环境变量，或换网络后重试。

2. **启动报缺少索引**  
   先完成「3. 构建向量索引」，确认 `dataset/ecommerce.index` 与 `dataset/index_ids.json` 存在。

3. **图片裂图**  
   保证 `dataset/images` 中文件与 `metadata.json` 里 `image_file` 等字段一致；静态资源挂载路径见 `main.py` 中 `IMAGES_URL_PREFIX`。

---

## 许可证与第三方

模型与数据集请遵循各自上游许可证；本项目代码仅供学习与研究使用。
