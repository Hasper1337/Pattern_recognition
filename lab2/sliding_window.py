# -*- coding: utf-8 -*-
"""
ЛР №2, вариант 1: поиск гипотез методом Sliding window + классификация CNN.

Шаги для каждого изображения:
  1. Sliding window: вырезаем все прямоугольники нескольких размеров с шагом.
  2. Эвристика: однотонные окна (малое СКО яркости) отсеиваются сразу -> Negative.
     Остальные — Positive, они идут на классификацию.
  3. Классификация гипотез предобученной CNN (MobileNetV3, ImageNet):
     вероятность "кот" = сумма вероятностей классов ImageNet 281–285
     (tabby, tiger cat, Persian, Siamese, Egyptian cat).
  4. Реальный класс окна: есть объект, если IoU окна с эталонной рамкой >= 0.5
     (эталонные рамки — gt.json, размечаются скриптом annotate.py).
  5. Таблица TP / FN / FP / TN и Accuracy = (TP + TN) / (P + N).

Запуск:
    pip install torch torchvision opencv-python
    python annotate.py --images recog_2 --out gt.json      # один раз
    python sliding_window.py --images recog_2 --gt gt.json
"""

import argparse
import json
import os

import cv2
import numpy as np

SCALES = (0.3, 0.45, 0.6, 0.8)   # сторона окна как доля меньшей стороны картинки
RATIOS = (1.0, 1.5)              # отношение ширины к высоте
STEP = 0.25                      # шаг = 1/4 размера окна
STD_THR = 20                     # эвристика: окна с СКО яркости меньше — фон
CAT_THR = 0.5                    # порог вероятности "кот"
IOU_THR = 0.5                    # порог IoU для "в окне есть объект"
CAT_CLASSES = list(range(281, 286))


# ---------------------------------------------------------------- гипотезы

def sliding_windows(h, w):
    """Все окна (x1, y1, x2, y2) по всем масштабам и пропорциям."""
    boxes = []
    for s in SCALES:
        for r in RATIOS:
            bh = int(min(h, w) * s)
            bw = min(int(bh * r), w)
            sy, sx = max(1, int(bh * STEP)), max(1, int(bw * STEP))
            for y in range(0, h - bh + 1, sy):
                for x in range(0, w - bw + 1, sx):
                    boxes.append((x, y, x + bw, y + bh))
    return boxes


def passes_heuristic(gray, box):
    x1, y1, x2, y2 = box
    return gray[y1:y2, x1:x2].std() >= STD_THR


# ---------------------------------------------------------------- классификатор

def make_classifier():
    """Возвращает функцию: список BGR-фрагментов -> массив вероятностей 'кот'."""
    import torch
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def classify(crops, batch=64):
        probs = []
        for i in range(0, len(crops), batch):
            x = np.stack([cv2.resize(c, (224, 224))[:, :, ::-1] for c in crops[i:i + batch]])
            x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float() / 255
            p = torch.softmax(model((x - mean) / std), dim=1)
            probs.append(p[:, CAT_CLASSES].sum(dim=1).numpy())
        return np.concatenate(probs) if probs else np.array([])

    return classify


# ---------------------------------------------------------------- метрики

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def process(path, gt_boxes, classify, out_dir):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = sliding_windows(*gray.shape)

    # шаг 1: эвристический отсев
    keep = [b for b in boxes if passes_heuristic(gray, b)]
    keep_set = set(keep)

    # шаг 2: классификация оставшихся гипотез
    probs = classify([img[y1:y2, x1:x2] for (x1, y1, x2, y2) in keep])
    is_cat = {b: p >= CAT_THR for b, p in zip(keep, probs)}

    # шаг 3: таблица
    t = dict(TP=0, FN=0, FP=0, TN=0)
    vis = img.copy()
    for b in boxes:
        pred = b in keep_set and is_cat[b]
        actual = any(iou(b, g) >= IOU_THR for g in gt_boxes)
        key = ("T" if pred == actual else "F") + ("P" if pred else "N")
        t[key] += 1
        if pred:
            cv2.rectangle(vis, b[:2], b[2:], (0, 0, 255) if actual else (0, 165, 255), 1)
    for g in gt_boxes:
        cv2.rectangle(vis, tuple(g[:2]), tuple(g[2:]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), vis)

    return dict(image=os.path.basename(path), hypotheses=len(boxes),
                positive=len(keep), negative=len(boxes) - len(keep),
                classified_cat=int(sum(is_cat.values())), **t)


def print_table(t, title):
    total = sum(t[k] for k in ("TP", "FN", "FP", "TN"))
    acc = (t["TP"] + t["TN"]) / total if total else 0
    print(f"\n{title}")
    print(f"{'':18}{'Predicted P':>13}{'Predicted N':>13}")
    print(f"{'Actual P':18}{t['TP']:>13}{t['FN']:>13}")
    print(f"{'Actual N':18}{t['FP']:>13}{t['TN']:>13}")
    print(f"Accuracy = (TP+TN)/(P+N) = ({t['TP']}+{t['TN']})/{total} = {acc:.4f}")
    return acc


def main(classify=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="recog_2")
    ap.add_argument("--gt", default="gt.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    classify = classify or make_classifier()

    rows = []
    for name in sorted(gt):
        r = process(os.path.join(args.images, name), gt[name], classify, args.out)
        rows.append(r)
        print(f"\n=== {name}: гипотез {r['hypotheses']}, на классификацию (Positive) {r['positive']}, "
              f"отсеяно (Negative) {r['negative']}, классификатор нашёл кота в {r['classified_cat']}")
        print_table(r, "Таблица для изображения")

    total = {k: sum(r[k] for r in rows) for k in ("TP", "FN", "FP", "TN")}
    print_table(total, "===== ИТОГО по всем изображениям =====")
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nКартинки с рамками: {args.out}/ (зелёный — эталон, красный — TP, оранжевый — FP)")


if __name__ == "__main__":
    main()
