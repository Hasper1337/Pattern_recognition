# -*- coding: utf-8 -*-
"""
ЛР №5, шаг 4: дообучение предобученной YOLOv8n на своём объекте.

Запуск (лучше в Google Colab с GPU):
    python train.py --epochs 50
Лучшие веса: runs/detect/<name>/weights/best.pt
"""

import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data.yaml")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--name", default="my_object")
args = ap.parse_args()

model = YOLO("yolov8n.pt")
model.train(data=args.data, epochs=args.epochs, imgsz=640, name=args.name, exist_ok=True)

metrics = model.val(data=args.data, split="test")
print(f"test: mAP50 = {metrics.box.map50:.3f}, mAP50-95 = {metrics.box.map:.3f}")
print(f"Веса: runs/detect/{args.name}/weights/best.pt")
