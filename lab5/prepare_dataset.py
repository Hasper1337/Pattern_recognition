# -*- coding: utf-8 -*-
"""
ЛР №5, шаг 3: разбиение на train / val / test + аугментация обучающей выборки.

На вход — экспорт Label Studio в формате YOLO (папки images/ и labels/,
файл classes.txt). На выход — структура, которую ждёт YOLO:

    dataset/images/{train,val,test}   dataset/labels/{train,val,test}   data.yaml

Аугментация (часть предобработки из шага 1, варианта 1) — только для train:
  * горизонтальное отражение (x_center -> 1 - x_center);
  * случайное изменение яркости/контраста.

Запуск:
    python prepare_dataset.py --src labelstudio_export --dst dataset
"""

import argparse
import os
import random
import shutil

import cv2
import numpy as np


def read_labels(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def write_labels(path, rows):
    with open(path, "w") as f:
        f.writelines(" ".join(r) + "\n" for r in rows)


def augment(img, rows, rng):
    """Возвращает список (картинка, разметка, суффикс)."""
    flipped = cv2.flip(img, 1)
    flipped_rows = [[c, f"{1 - float(x):.6f}", y, w, h] for c, x, y, w, h in rows]
    alpha, beta = rng.uniform(0.7, 1.3), rng.uniform(-30, 30)
    bright = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return [(flipped, flipped_rows, "flip"), (bright, rows, "bright")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="labelstudio_export")
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    images = sorted(n for n in os.listdir(os.path.join(args.src, "images"))
                    if n.lower().endswith((".jpg", ".jpeg", ".png")))
    rng.shuffle(images)
    n_test, n_val = int(len(images) * args.test), int(len(images) * args.val)
    splits = {"test": images[:n_test], "val": images[n_test:n_test + n_val], "train": images[n_test + n_val:]}

    for split, names in splits.items():
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)
        for name in names:
            stem = os.path.splitext(name)[0]
            src_img = os.path.join(args.src, "images", name)
            src_lbl = os.path.join(args.src, "labels", stem + ".txt")
            rows = read_labels(src_lbl)
            shutil.copy(src_img, os.path.join(args.dst, "images", split, name))
            write_labels(os.path.join(args.dst, "labels", split, stem + ".txt"), rows)
            if split == "train":
                img = cv2.imread(src_img)
                for aug_img, aug_rows, suffix in augment(img, rows, rng):
                    cv2.imwrite(os.path.join(args.dst, "images", split, f"{stem}_{suffix}.jpg"), aug_img)
                    write_labels(os.path.join(args.dst, "labels", split, f"{stem}_{suffix}.txt"), aug_rows)
        print(f"{split}: {len(names)} изображений" + (" (+ аугментация x2)" if split == "train" else ""))

    classes_path = os.path.join(args.src, "classes.txt")
    names = [l.strip() for l in open(classes_path, encoding="utf-8") if l.strip()] if os.path.exists(classes_path) else ["object"]
    dst = os.path.abspath(args.dst)
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {dst}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                f"nc: {len(names)}\nnames: {names}\n")
    print("Создан data.yaml, классы:", names)


if __name__ == "__main__":
    main()
