# -*- coding: utf-8 -*-
"""
模型模块：定义用于 CIFAR-10 的卷积神经网络（CNN）。

算法要点：
- 卷积层：局部感受野 + 权值共享，自动学习边缘、纹理等层次特征。
- 批归一化（Batch Normalization）：稳定训练、加速收敛。
- 最大池化（Max Pooling）：降维并增强平移不变性。
- Dropout：训练时随机丢弃部分神经元，减轻过拟合。
- 全连接 + Softmax：将高层特征映射为 10 类概率分布。
"""

from __future__ import annotations

import tensorflow as tf

import config


def build_cnn(
    input_shape: tuple[int, int, int] = (
        config.IMG_HEIGHT,
        config.IMG_WIDTH,
        config.IMG_CHANNELS,
    ),
    num_classes: int = config.NUM_CLASSES,
) -> tf.keras.Model:
    """搭建轻量级 CNN，适合课程作业在 CPU 上数分钟到数十分钟完成训练。"""
    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(
        x
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model
