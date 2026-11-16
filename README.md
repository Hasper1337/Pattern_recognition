# Распознавание образов — ЛР 1–5, вариант 1

```bash
pip install -r requirements.txt
```

Картинки с прошлых лаб (10 штук с котами) положите в папку `recog_2/` внутри нужной лабы
или передайте путь параметром `--images`. Для пробы в `images/` лежат 5 демо-картинок.

| ЛР | Тема варианта 1 | Что запускать |
|---|---|---|
| 1 | Фильтр Собеля | `lab1/sobel.py` + текст отчёта `lab1/report.md` |
| 2 | Sliding window | `lab2/annotate.py` → `lab2/sliding_window.py` |
| 3 | YOLO (задание 1) + SSD (задание 2) | `lab3/yolo_compare.py`, `lab3/ssd.md` |
| 4 | DeepLab | `lab4/deeplab_segment.py` |
| 5 | Трекинг, замена шага 1 (предобработка) | см. порядок ниже |

## ЛР 1. Собель

```bash
cd lab1
python sobel.py --image ../images/1_cat.jpg --out results
```

Своя реализация на NumPy сравнивается с `cv2.Sobel` и `skimage.filters.sobel`, картинки для отчёта сохраняются в `results/`. В `report.md` — разделы отчёта по заданию (автор и ссылка, математика, библиотеки, пример, рекомендации); титульный лист оформите по методичке.

## ЛР 2. Sliding window

```bash
cd lab2
python annotate.py --images recog_2 --out gt.json          # обвести котов мышью, один раз
python sliding_window.py --images recog_2 --gt gt.json
```

- Гипотезы: окна 4 размеров × 2 пропорций с шагом 1/4 окна.
- Отсев эвристикой (однотонные окна) → Negative, остальные → Positive, идут в классификатор.
- Классификатор: предобученная MobileNetV3 (ImageNet), «кот» = сумма вероятностей классов 281–285.
- Реальный класс окна: IoU с эталонной рамкой ≥ 0.5.
- Для каждой картинки и в сумме печатается таблица TP/FN/FP/TN и Accuracy.

Параметры (размеры окон, пороги) — константы в начале файла.

## ЛР 3. YOLO и SSD

```bash
cd lab3
python yolo_compare.py --images ../lab2/recog_2                 # быстро, coco8
python yolo_compare.py --data VOC.yaml --epochs 20 --images ../lab2/recog_2   # в Colab с GPU
```

Сравниваются: (1) новая модель, обученная с нуля, (2) предобученная, (3) предобученная + дообученная. Таблица — `runs_compare/results.md`. `ssd.md` — улучшенное описание SSD для отправки преподавателю.

## ЛР 4. DeepLab

```bash
cd lab4
python deeplab_segment.py --images ../lab2/recog_2 --out results
python deeplab_segment.py --images ../lab2/recog_2 --model mobilenet   # если медленно на CPU
```

Для каждой картинки: маска классов, наложение на исходник и доли площади найденных классов (cat, person, …).

## ЛР 5. Трекинг (вариант 1 — заменён шаг 1, предобработка)

Снимите 2–3 коротких видео с движущимся объектом (не монеткой) в `lab5/videos/`.

```bash
cd lab5
python check_pretrained.py --video videos/1.mp4    # шаг 0: предобученная YOLO не узнаёт объект
python preprocess.py --videos videos --out frames  # шаг 1 (ЗАМЕНЁН): кадры, очистка, CLAHE, 640x640
```

**Шаг 2 — разметка в Label Studio:** создайте проект *Object Detection with Bounding Boxes*
(разметка **изображений**, не видео), загрузите `frames/`, обведите объект на каждом кадре,
экспортируйте в формате **YOLO** и распакуйте в `labelstudio_export/` (там будут `images/`, `labels/`, `classes.txt`).

```bash
python prepare_dataset.py --src labelstudio_export --dst dataset   # шаг 3: train/val/test + аугментация
python train.py --epochs 50                                         # шаг 4: дообучение YOLOv8n
python track.py --weights runs/detect/my_object/weights/best.pt --video videos/2.mp4 --show   # шаги 5–6
```

**Что изменено в шаге 1 по сравнению с примером:**

| Было в примере | Стало |
|---|---|
| Предобработка пропущена | Берётся каждый N-й кадр, смазанные кадры (дисперсия лапласиана) и дубликаты отбрасываются |
| — | Выравнивание контраста CLAHE — объект лучше отделяется от фона |
| Resize до 640x640 только в шаге 3 | Resize сразу при извлечении кадров, датасет меньше |
| Аугментации нет | Отражение и изменение яркости для train (с пересчётом разметки) |
| При трекинге кадр только сжимается | При трекинге каждый кадр проходит ту же `preprocess_frame()` |

Результат трекинга — `tracked.mp4`; в консоли — число ID треков (в идеале — по одному на объект).
