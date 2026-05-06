# -*- coding: utf-8 -*-
"""
数据模块：下载并加载 CIFAR-10，做预处理与增强。

CIFAR-10 简介（开源学术数据集）：
- 加拿大高级研究计划所发布，广泛用于图像分类基准测试。
- 60,000 张 32x32 彩色图，10 类，每类 6,000 张；50k 训练 / 10k 测试。
- 可通过 keras.datasets.cifar10.load_data() 自动下载到本地缓存。
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def load_cifar10_raw():
    """从 Keras 加载原始 uint8 格式的训练集与测试集。"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def normalize_images(x: np.ndarray) -> np.ndarray:
    """将像素从 [0,255] 缩放到 [0,1] 浮点，利于神经网络稳定训练。"""
    return x.astype("float32") / 255.0


def labels_to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """把形状 (N,1) 的整数标签转为 one-hot 编码，配合 categorical_crossentropy。"""
    y = y.reshape(-1)
    return tf.keras.utils.to_categorical(y, num_classes)


def build_train_augmentation() -> tf.keras.Sequential:
    """
    轻量数据增强：随机水平翻转、小幅平移与对比度抖动。
    作用：扩大有效训练样本多样性，减轻过拟合（对小图像分类很常见）。
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="train_augmentation",
    )


def prepare_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    batch_size: int,
    validation_split: float,
    seed: int = 42,
):
    """
    归一化、划分验证集、构建 tf.data.Dataset 以支持 prefetch 与增强。

    返回：
        train_ds, val_ds, test_ds —— 均已启用 shuffle / batch / prefetch
    """
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    y_train_oh = labels_to_one_hot(y_train, num_classes)
    y_test_oh = labels_to_one_hot(y_test, num_classes)

    n = x_train.shape[0]
    val_n = int(n * validation_split)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    x_tr, y_tr = x_train[tr_idx], y_train_oh[tr_idx]
    x_val, y_val = x_train[val_idx], y_train_oh[val_idx]

    aug = build_train_augmentation()

    def apply_aug(x, y):
        return aug(x, training=True), y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(buffer_size=len(x_tr), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_oh))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds
