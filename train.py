# -*- coding: utf-8 -*-
"""
训练脚本：编译模型、训练、保存权重并绘制损失/准确率曲线。

优化算法：Adam（自适应学习率的一阶优化方法，对超参相对鲁棒）。
损失函数：分类交叉熵 categorical_crossentropy（多类单标签，标签为 one-hot）。
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf

import config
import data
import model


def parse_args():
    p = argparse.ArgumentParser(description="在 CIFAR-10 上训练 CNN")
    p.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--validation-split",
        type=float,
        default=config.DEFAULT_VALIDATION_SPLIT,
        help="从官方训练 50k 中划出一部分做验证",
    )
    p.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    """将训练/验证的 loss 与 accuracy 存成 PNG，便于写入实验报告。"""
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, h["loss"], label="train_loss")
    ax[0].plot(epochs, h["val_loss"], label="val_loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, h["accuracy"], label="train_acc")
    ax[1].plot(epochs, h["val_accuracy"], label="val_acc")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = data.load_cifar10_raw()
    train_ds, val_ds, test_ds = data.prepare_datasets(
        x_train,
        y_train,
        x_test,
        y_test,
        num_classes=config.NUM_CLASSES,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    m = model.build_cnn()
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    history = m.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb,
        verbose=1,
    )

    m.save(config.MODEL_PATH)
    plot_history(history, config.HISTORY_PLOT_PATH)

    test_loss, test_acc = m.evaluate(test_ds, verbose=0)
    print(f"测试集 loss={test_loss:.4f}, accuracy={test_acc:.4f}")
    print(f"模型已保存: {config.MODEL_PATH}")
    print(f"训练曲线已保存: {config.HISTORY_PLOT_PATH}")


if __name__ == "__main__":
    main()
