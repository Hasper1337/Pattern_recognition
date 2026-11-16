# -*- coding: utf-8 -*-
"""
ЛР №5, вариант 1 — ЗАМЕНЁННЫЙ ШАГ 1: предобработка данных.

В примере преподавателя шаг 1 пропущен. Здесь он реализован так:
  1. Из видео берётся каждый N-й кадр (соседние кадры почти одинаковы).
  2. Очистка: отбрасываются смазанные кадры (дисперсия лапласиана < порога)
     и почти дубликаты (с последнего сохранённого кадра почти ничего не изменилось).
  3. Улучшение: выравнивание контраста CLAHE по яркости (канал L в LAB)
     — объект лучше выделяется на фоне при плохом освещении.
  4. Уменьшение до 640x640 сразу (в примере это делалось только в шаге 3).
  (Аугментация — отражение и яркость — делается в prepare_dataset.py,
   потому что при отражении нужно пересчитать и разметку.)

Та же функция preprocess_frame() применяется в track.py к каждому кадру
видео, чтобы модель видела при работе такие же кадры, как при обучении.

Запуск:
    python preprocess.py --videos videos --out frames --every 3
"""

import argparse
import os

import cv2

SIZE = 640


def preprocess_frame(frame, size=SIZE):
    """CLAHE по яркости + resize до size x size. Вход и выход — BGR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract(video_path, out_dir, every, blur_thr, dup_thr):
    cap = cv2.VideoCapture(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    stats = dict(read=0, blurry=0, duplicate=0, saved=0)
    last, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stats["read"] += 1
        i += 1
        if (i - 1) % every:
            continue
        if sharpness(frame) < blur_thr:
            stats["blurry"] += 1
            continue
        small = cv2.cvtColor(cv2.resize(frame, (320, 320)), cv2.COLOR_BGR2GRAY)
        # дубликат: изменилось меньше dup_thr % пикселей (шум камеры отсекаем порогом 25)
        if last is not None and (cv2.absdiff(small, last) > 25).mean() * 100 < dup_thr:
            stats["duplicate"] += 1
            continue
        last = small
        cv2.imwrite(os.path.join(out_dir, f"{base}_{i:05d}.jpg"), preprocess_frame(frame))
        stats["saved"] += 1
    cap.release()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos", help="папка с вашими видео")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--every", type=int, default=3, help="брать каждый N-й кадр")
    ap.add_argument("--blur_thr", type=float, default=30.0, help="порог резкости (меньше — кадр смазан)")
    ap.add_argument("--dup_thr", type=float, default=0.1, help="мин. %% изменившихся пикселей, иначе кадр — дубликат")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in sorted(os.listdir(args.videos)):
        if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            s = extract(os.path.join(args.videos, name), args.out, args.every, args.blur_thr, args.dup_thr)
            print(f"{name}: прочитано {s['read']}, смазанных {s['blurry']}, "
                  f"дубликатов {s['duplicate']}, сохранено {s['saved']}")
    print(f"Кадры для разметки: {args.out}/")


if __name__ == "__main__":
    main()
