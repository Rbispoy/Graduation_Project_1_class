# -*- coding: utf-8 -*-
"""全局配置：数据集元信息、路径与训练超参数。"""

import os

# 项目根目录（本文件所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练得到的模型与准确率曲线保存路径
MODEL_DIR = os.path.join(ROOT_DIR, "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn.keras")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")

# CIFAR-10 官方类别名称（与标签 0~9 一一对应）
CLASS_NAMES = (
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车",
)

# 图像尺寸与通道数（CIFAR-10 固定为 32x32 RGB）
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
NUM_CLASSES = 10

# 训练默认超参数（可在命令行覆盖）
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 30
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_LEARNING_RATE = 1e-3
