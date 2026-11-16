# -*- coding: utf-8 -*-
"""
ЛР №5, шаг 0: показать, что предобученная YOLOv8 не отслеживает наш объект
(не находит его или присваивает неверный класс).

Запуск:
    python check_pretrained.py --video videos/1.mp4
Результат: runs/detect/pretrained/<видео> с рамками + в консоли — какие классы видит модель.
"""

import argparse
from collections import Counter

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--conf", type=float, default=0.25)
args = ap.parse_args()

model = YOLO("yolov8n.pt")
results = model.predict(source=args.video, conf=args.conf, save=True, name="pretrained", stream=True, verbose=False)

counter, frames, empty = Counter(), 0, 0
for r in results:
    frames += 1
    if len(r.boxes) == 0:
        empty += 1
    counter.update(model.names[int(c)] for c in r.boxes.cls)

print(f"Кадров: {frames}, кадров без детекций: {empty}")
print("Что видит предобученная модель (класс: число рамок):", dict(counter.most_common()))
