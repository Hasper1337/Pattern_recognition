# -*- coding: utf-8 -*-
"""
ЛР №5, шаги 5–6: обнаружение дообученной YOLOv8 + трекинг DeepSORT.

Каждый кадр проходит ту же предобработку, что и обучающие данные
(preprocess_frame: CLAHE + 640x640), рамки масштабируются обратно
в размер исходного кадра и передаются в DeepSORT, который присваивает ID.

Запуск:
    pip install ultralytics deep-sort-realtime
    python track.py --weights runs/detect/my_object/weights/best.pt --video videos/2.mp4
    python track.py ... --show        # показывать видео в окне во время обработки
"""

import argparse

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from preprocess import SIZE, preprocess_frame

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="tracked.mp4")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--show", action="store_true")
args = ap.parse_args()

model = YOLO(args.weights)
tracker = DeepSort(max_age=70, n_init=3, max_cosine_distance=0.7)

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
sx, sy = W / SIZE, H / SIZE
ids = set()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(preprocess_frame(frame), conf=args.conf, verbose=False)[0]
    detections = []
    for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                           result.boxes.cls.tolist()):
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy           # обратно в размер кадра
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))  # формат DeepSORT: ltwh

    for track in tracker.update_tracks(detections, frame=frame):
        if not track.is_confirmed():
            continue
        ids.add(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)
    if args.show:
        cv2.imshow("tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC — выход
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Готово: {args.out}. Всего ID треков: {len(ids)} (в идеале — по одному на объект)")
