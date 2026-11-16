# -*- coding: utf-8 -*-
"""
ЛР №4, вариант 1: семантическая сегментация моделью DeepLab (DeepLabV3, torchvision).

Модель предобучена на подмножестве COCO с 20 классами Pascal VOC + фон.
Для каждого изображения из папки (набор с прошлых лаб) скрипт:
  * строит маску классов (каждому пикселю — класс);
  * сохраняет цветную маску и её наложение на исходник;
  * печатает найденные классы и их долю площади.

Запуск:
    pip install torch torchvision opencv-python matplotlib
    python deeplab_segment.py --images recog_2 --out results
    python deeplab_segment.py --images recog_2 --model mobilenet   # быстрее на CPU
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import segmentation as seg

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# фиксированная палитра: фон — чёрный, остальные — разные цвета (RGB)
PALETTE = np.array([[0, 0, 0]] + [
    [(37 * i) % 256, (97 * i + 80) % 256, (173 * i + 40) % 256] for i in range(1, 21)
], dtype=np.uint8)


def load_model(name):
    if name == "mobilenet":
        weights = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = seg.deeplabv3_mobilenet_v3_large(weights=weights)
    else:
        weights = seg.DeepLabV3_ResNet50_Weights.DEFAULT
        model = seg.deeplabv3_resnet50(weights=weights)
    return model.eval(), weights.transforms()


@torch.no_grad()
def segment(model, transform, img_rgb):
    """Возвращает маску классов (H x W) в размере исходного изображения."""
    x = transform(torch.from_numpy(img_rgb).permute(2, 0, 1)).unsqueeze(0)
    out = model(x)["out"]                                         # 1 x 21 x h x w
    out = torch.nn.functional.interpolate(out, size=img_rgb.shape[:2], mode="bilinear", align_corners=False)
    return out.argmax(1)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="recog_2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", choices=["resnet50", "mobilenet"], default="resnet50")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, transform = load_model(args.model)
    names = sorted(n for n in os.listdir(args.images) if n.lower().endswith((".jpg", ".jpeg", ".png")))

    for name in names:
        img = cv2.cvtColor(cv2.imread(os.path.join(args.images, name)), cv2.COLOR_BGR2RGB)
        mask = segment(model, transform, img)
        color = PALETTE[mask]
        overlay = (0.5 * img + 0.5 * color).astype(np.uint8)
        overlay[mask == 0] = img[mask == 0]                       # фон не закрашиваем

        # статистика по классам
        ids, counts = np.unique(mask, return_counts=True)
        found = [(VOC_CLASSES[i], c / mask.size) for i, c in zip(ids, counts) if i != 0]
        print(f"{name}: " + (", ".join(f"{c} {p:.1%}" for c, p in found) or "объектов не найдено"))

        # картинка: исходник | маска | наложение + легенда
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for a, im, t in zip(ax, (img, color, overlay), ("Исходное", "Маска DeepLabV3", "Наложение")):
            a.imshow(im); a.set_title(t); a.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i] / 255) for i in ids if i != 0]
        labels = [f"{VOC_CLASSES[i]}" for i in ids if i != 0]
        if handles:
            ax[1].legend(handles, labels, loc="lower right", fontsize=9)
        plt.tight_layout()
        base = os.path.splitext(name)[0]
        plt.savefig(os.path.join(args.out, f"{base}_seg.png"), dpi=100)
        plt.close(fig)
        cv2.imwrite(os.path.join(args.out, f"{base}_mask.png"), mask.astype(np.uint8))  # номера классов

    print(f"Результаты: {args.out}/")


if __name__ == "__main__":
    main()
