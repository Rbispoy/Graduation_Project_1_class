# -*- coding: utf-8 -*-
"""
预测脚本：加载已训练模型，在测试集上评估，并可对单张图片推理演示。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

import config
import data


def parse_args():
    p = argparse.ArgumentParser(description="使用已保存模型对 CIFAR-10 或自定义图片推理")
    p.add_argument(
        "--model",
        default=config.MODEL_PATH,
        help=".keras 模型路径",
    )
    p.add_argument("--image", default=None, help="可选：本地图片路径 (RGB)，将 resize 到 32x32")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def evaluate_test(model_path: str, batch_size: int) -> None:
    """在官方测试集上计算准确率（标签为 one-hot，与训练一致）。"""
    (_, _), (x_test, y_test) = data.load_cifar10_raw()
    x_test = data.normalize_images(x_test)
    y_test_oh = data.labels_to_one_hot(y_test, config.NUM_CLASSES)
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_oh))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    m = tf.keras.models.load_model(model_path)
    loss, acc = m.evaluate(test_ds, verbose=0)
    print(f"测试集 loss={loss:.4f}, accuracy={acc:.4f}")


def load_image_for_model(image_path: str) -> np.ndarray:
    """读取本地图片并缩放到模型输入尺寸，返回 float32 数组 (H, W, 3)，数值范围 [0,1]。"""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
    return np.asarray(img, dtype="float32") / 255.0


def predict_top_k(
    model: tf.keras.Model,
    image_path: str,
    k: int = 5,
) -> list[tuple[str, int, float]]:
    """
    返回 Top-k 结果列表：(类别中文名, 类别索引, 概率)。
    供命令行与图形界面共用。
    """
    arr = load_image_for_model(image_path)
    batch = np.expand_dims(arr, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1][:k]
    out = []
    for idx in order:
        i = int(idx)
        out.append((config.CLASS_NAMES[i], i, float(probs[i])))
    return out


def predict_one_image(model: tf.keras.Model, image_path: str) -> None:
    """读取本地图片，预处理后用模型输出 Top-3 类别。"""
    results = predict_top_k(model, image_path, k=3)
    print("Top-3 预测：")
    for rank, (name, idx, p) in enumerate(results, start=1):
        print(f"  {rank}. {name} (索引 {idx}), 概率 {p:.4f}")


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"找不到模型文件: {args.model}，请先运行 train.py 训练")

    if args.image:
        m = tf.keras.models.load_model(args.model)
        predict_one_image(m, args.image)
    else:
        evaluate_test(args.model, args.batch_size)


if __name__ == "__main__":
    main()
