# -*- coding: utf-8 -*-
"""
ЛР №1, вариант 1: фильтр (оператор) Собеля.

Что делает скрипт:
  1. Своя реализация оператора Собеля на NumPy (свёртка 3x3 вручную).
  2. Готовые реализации: OpenCV (cv2.Sobel) и scikit-image (skimage.filters.sobel).
  3. Сравнение своей реализации с OpenCV (максимальная разница).
  4. Сохраняет картинки для отчёта: Gx, Gy, модуль градиента, направление,
     бинарные границы и сравнение с библиотеками.

Запуск:
    python sobel.py --image ../images/1_cat.jpg --out results
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel as sk_sobel

# Ядра Собеля
KX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float64)
KY = KX.T  # [[-1,-2,-1],[0,0,0],[1,2,1]]


def convolve3x3(img, kernel):
    """Корреляция с ядром 3x3 (как в cv2.filter2D), края — отражение."""
    padded = np.pad(img, 1, mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    h, w = img.shape
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def sobel_numpy(gray):
    """Возвращает Gx, Gy, модуль градиента G и направление theta (в градусах)."""
    gray = gray.astype(np.float64)
    gx = convolve3x3(gray, KX)
    gy = convolve3x3(gray, KY)
    g = np.hypot(gx, gy)                      # G = sqrt(Gx^2 + Gy^2)
    theta = np.degrees(np.arctan2(gy, gx))    # направление градиента
    return gx, gy, g, theta


def to_uint8(a):
    a = np.abs(a)
    return (255 * a / a.max()).astype(np.uint8) if a.max() > 0 else a.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="../images/1_cat.jpg")
    ap.add_argument("--out", default="results")
    ap.add_argument("--blur", type=int, default=3, help="размер гауссова размытия перед Собелем (0 — без)")
    ap.add_argument("--thr", type=float, default=0.2, help="порог для бинарных границ (доля от max G)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.blur:
        gray = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)  # подавляем шум

    # 1) своя реализация
    gx, gy, g, theta = sobel_numpy(gray)
    edges = (g > args.thr * g.max()).astype(np.uint8) * 255

    # 2) OpenCV
    cv_gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    cv_gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    cv_g = np.hypot(cv_gx, cv_gy)

    # 3) scikit-image (возвращает нормированный модуль градиента)
    sk_g = sk_sobel(gray.astype(np.float64) / 255)

    diff = np.abs(g - cv_g).max()
    print(f"Изображение: {args.image}, размер {gray.shape[1]}x{gray.shape[0]}")
    print(f"Макс. |G| своя реализация: {g.max():.1f}, OpenCV: {cv_g.max():.1f}")
    print(f"Максимальная разница своя vs OpenCV: {diff:.6f}")
    print(f"Доля пикселей-границ при пороге {args.thr}: {(edges > 0).mean():.3f}")

    # --- картинки для отчёта
    cv2.imwrite(os.path.join(args.out, "gx.png"), to_uint8(gx))
    cv2.imwrite(os.path.join(args.out, "gy.png"), to_uint8(gy))
    cv2.imwrite(os.path.join(args.out, "magnitude.png"), to_uint8(g))
    cv2.imwrite(os.path.join(args.out, "edges.png"), edges)

    # направление градиента цветом (оттенок = угол, яркость = модуль)
    hsv = np.zeros((*gray.shape, 3), np.uint8)
    hsv[..., 0] = ((theta + 180) / 2).astype(np.uint8)   # 0..180 для OpenCV
    hsv[..., 1] = 255
    hsv[..., 2] = to_uint8(g)
    cv2.imwrite(os.path.join(args.out, "direction.png"), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Исходное"),
        (np.abs(gx), "Gx (вертикальные границы)"),
        (np.abs(gy), "Gy (горизонтальные границы)"),
        (g, "|G| — своя реализация"),
        (cv_g, "|G| — cv2.Sobel"),
        (sk_g, "|G| — skimage.filters.sobel"),
    ]
    for a, (im, title) in zip(ax.ravel(), panels):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray")
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "comparison.png"), dpi=120)
    print(f"Картинки сохранены в {args.out}/")


if __name__ == "__main__":
    main()
