# -*- coding: utf-8 -*-
"""
Разметка эталонных рамок (ground truth) для ЛР №2.

Для каждой картинки открывается окно: выделите мышью каждого кота
(ENTER/SPACE — подтвердить рамку, ещё раз ENTER/ESC — перейти к следующей картинке).
Результат: gt.json вида {"1.jpg": [[x1, y1, x2, y2], ...], ...}

Запуск:
    python annotate.py --images recog_2 --out gt.json
"""

import argparse
import json
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="recog_2")
    ap.add_argument("--out", default="gt.json")
    args = ap.parse_args()

    gt = {}
    if os.path.exists(args.out):                  # можно доразмечать
        with open(args.out, encoding="utf-8") as f:
            gt = json.load(f)

    names = sorted(n for n in os.listdir(args.images) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    for name in names:
        if name in gt:
            continue
        img = cv2.imread(os.path.join(args.images, name))
        rois = cv2.selectROIs(f"{name}: выделите объекты, ESC - дальше", img, showCrosshair=False)
        cv2.destroyAllWindows()
        gt[name] = [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in rois]
        print(name, gt[name])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
