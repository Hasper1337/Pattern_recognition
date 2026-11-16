# -*- coding: utf-8 -*-
"""
ЛР №3, задание 1: сравнение точности обнаружения YOLO в трёх случаях
  1) новая модель, обученная с нуля        (yolo11n.yaml -> train)
  2) предобученная модель без дообучения   (yolo11n.pt   -> val)
  3) предобученная модель, дообученная     (yolo11n.pt   -> train -> val)

Установка:   pip install ultralytics      (долго, тянет PyTorch)
Запуск:      python yolo_compare.py                      # быстрый вариант на coco8
             python yolo_compare.py --data VOC.yaml --epochs 50 --imgsz 640
             python yolo_compare.py --images recog_2     # + показ детекций на своих картинках

Результаты: таблица в консоли, runs_compare/results.csv и results.md,
            картинки с рамками в runs_compare/predict_*/.

ВАЖНО про честность сравнения:
  * coco8 — это 8 картинок (4 train / 4 val), поэтому цифры шумные, а предобученная
    модель уже видела COCO при обучении. Для содержательного вывода используйте
    --data VOC.yaml (или свой датасет) — но это часы обучения; в Colab с GPU.
  * Все три модели оцениваются на ОДНОМ И ТОМ ЖЕ val-наборе, с одинаковым imgsz.
"""

import argparse
import csv
import os

from ultralytics import YOLO

PROJECT = "runs_compare"


def evaluate(model, data, imgsz, tag):
    """Считает метрики на val-наборе и возвращает словарь."""
    m = model.val(data=data, imgsz=imgsz, project=PROJECT, name=f"val_{tag}", exist_ok=True, verbose=False)
    return {
        "model": tag,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
    }


def predict_folder(model, folder, tag, conf=0.25):
    """Качественное сравнение: детекции на своих картинках (например, коты из ЛР №2)."""
    paths = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not paths:
        print(f"[{tag}] в {folder} нет изображений")
        return
    results = model.predict(paths, conf=conf, save=True, project=PROJECT, name=f"predict_{tag}", exist_ok=True, verbose=False)
    total = sum(len(r.boxes) for r in results)
    print(f"[{tag}] {len(paths)} изображений, найдено объектов: {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coco8.yaml", help="coco8.yaml | VOC.yaml | путь к своему yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolo11n", help="размер модели: yolo11n / yolo11s / ...")
    ap.add_argument("--images", default=None, help="папка с картинками для качественной проверки (recog_2)")
    ap.add_argument("--device", default=None, help="0 для GPU, cpu для процессора; по умолчанию авто")
    args = ap.parse_args()

    train_kw = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, project=PROJECT, exist_ok=True)

    rows, models = [], {}

    # 1) Новая модель, обучение с нуля (случайные веса, только архитектура из .yaml)
    scratch = YOLO(f"{args.weights}.yaml")
    scratch.train(name="train_scratch", pretrained=False, **train_kw)
    models["1_scratch"] = scratch                      # после train в модели лежат лучшие веса
    rows.append(evaluate(scratch, args.data, args.imgsz, "1_scratch (обучена с нуля)"))

    # 2) Предобученная (COCO) модель без дообучения
    pretrained = YOLO(f"{args.weights}.pt")
    models["2_pretrained"] = pretrained
    rows.append(evaluate(pretrained, args.data, args.imgsz, "2_pretrained (без дообучения)"))

    # 3) Предобученная модель, дообученная на тех же данных
    finetuned = YOLO(f"{args.weights}.pt")
    finetuned.train(name="train_finetune", **train_kw)
    models["3_finetuned"] = finetuned
    rows.append(evaluate(finetuned, args.data, args.imgsz, "3_finetuned (дообучена)"))

    # --- сводная таблица
    os.makedirs(PROJECT, exist_ok=True)
    keys = ["model", "precision", "recall", "mAP50", "mAP50-95"]
    with open(os.path.join(PROJECT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = ["| Модель | Precision | Recall | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} | {r['mAP50-95']:.3f} |")
    table = "\n".join(md)
    with open(os.path.join(PROJECT, "results.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)

    # --- качественное сравнение на своих картинках
    if args.images:
        for tag, m in models.items():
            predict_folder(m, args.images, tag)


if __name__ == "__main__":
    main()
