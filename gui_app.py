# -*- coding: utf-8 -*-
"""
简易图形界面：选择本地图片，调用已训练的 CIFAR-10 模型显示 Top 类别概率。

依赖：tkinter（Python 自带）、Pillow、TensorFlow；无需浏览器或额外前端框架。
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import config
import predict


def main():
    root = tk.Tk()
    root.title("CIFAR-10 图像识别（简易界面）")
    root.minsize(520, 420)

    status_var = tk.StringVar(value="正在加载模型，请稍候…")
    ttk.Label(root, textvariable=status_var, padding=8).pack(fill=tk.X)

    preview_frame = ttk.LabelFrame(root, text="预览（缩放显示）", padding=8)
    preview_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    preview_label = ttk.Label(preview_frame)
    preview_label.pack()

    result_frame = ttk.LabelFrame(root, text="识别结果（Top 5）", padding=8)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    tree = ttk.Treeview(
        result_frame,
        columns=("rank", "name", "prob"),
        show="headings",
        height=5,
    )
    tree.heading("rank", text="排名")
    tree.heading("name", text="类别")
    tree.heading("prob", text="概率")
    tree.column("rank", width=50, anchor=tk.CENTER)
    tree.column("name", width=120, anchor=tk.W)
    tree.column("prob", width=100, anchor=tk.E)
    tree.pack(fill=tk.BOTH, expand=True)

    model_holder: list = [None]
    photo_holder: list = [None]

    def fail(msg: str):
        status_var.set("错误")
        messagebox.showerror("错误", msg)

    def load_model_bg():
        if not os.path.isfile(config.MODEL_PATH):
            root.after(
                0,
                lambda: fail(
                    f"未找到模型文件：\n{config.MODEL_PATH}\n\n请先在本目录运行：python train.py"
                ),
            )
            return
        try:
            import tensorflow as tf

            m = tf.keras.models.load_model(config.MODEL_PATH)
        except Exception as e:
            err_msg = str(e)
            root.after(0, lambda: fail(f"加载模型失败：{err_msg}"))
            return
        model_holder[0] = m
        root.after(0, lambda: status_var.set("就绪 — 请点击按钮选择一张图片"))

    threading.Thread(target=load_model_bg, daemon=True).start()

    def on_pick_file():
        path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("JPEG", "*.jpeg"),
                ("其它", "*.bmp;*.gif;*.webp"),
                ("全部文件", "*.*"),
            ],
        )
        if not path:
            return
        status_var.set(f"识别中：{os.path.basename(path)}…")

        def work():
            m = model_holder[0]
            if m is None:
                root.after(0, lambda: messagebox.showwarning("请稍候", "模型仍在加载中。"))
                root.after(0, lambda: status_var.set("等待模型加载…"))
                return
            try:
                results = predict.predict_top_k(m, path, k=5)
            except Exception as e:
                err = str(e)

                def show_err():
                    messagebox.showerror("识别失败", err)
                    status_var.set("就绪")

                root.after(0, show_err)
                return

            def apply_ui():
                for row in tree.get_children():
                    tree.delete(row)
                for rank, (name, _idx, p) in enumerate(results, start=1):
                    tree.insert("", tk.END, values=(rank, name, f"{p:.2%}"))
                try:
                    img = Image.open(path).convert("RGB")
                    img.thumbnail((280, 280), Image.Resampling.LANCZOS)
                    photo_holder[0] = ImageTk.PhotoImage(img)
                    preview_label.configure(image=photo_holder[0])
                except Exception:
                    preview_label.configure(image="")
                status_var.set(f"完成：{os.path.basename(path)}")

            root.after(0, apply_ui)

        threading.Thread(target=work, daemon=True).start()

    btn = ttk.Button(root, text="选择图片并识别", command=on_pick_file)
    btn.pack(pady=8)

    root.mainloop()


if __name__ == "__main__":
    main()
