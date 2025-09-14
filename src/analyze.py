# from ultralytics import YOLO
# from PIL import Image
# import argparse
#
# def analyze(image_path: str, model_path: str):
#     model = YOLO(model_path)
#     img = Image.open(image_path)
#     img_area = img.width * img.height
#
#     # Предсказание
#     results = model.predict(image_path, conf=0.1, imgsz=640, save=False)
#     r = results[0]
#
#     # Словарь для подсчёта площадей каждого класса
#     class_areas = {name: 0 for name in model.names.values()}
#
#     for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
#         x1, y1, x2, y2 = box.tolist()
#         area = (x2 - x1) * (y2 - y1)
#         cls_name = model.names[int(cls_id)]
#         class_areas[cls_name] = class_areas.get(cls_name, 0) + area
#
#     print(f"\n📊 Анализ изображения: {image_path}")
#     for cls_name, area in class_areas.items():
#         percent = (area / img_area) * 100
#         print(f"  🏷 {cls_name}: {percent:.2f}% площади изображения")
#
#     print("\n✅ Анализ завершён")
#
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--image", required=True, help="Путь к изображению")
#     ap.add_argument("--model", required=True, help="Путь к модели YOLO (.pt)")
#     args = ap.parse_args()
#
#     analyze(args.image, args.model)



from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import argparse, json
from pathlib import Path

def analyze(image_path: str, model_path: str):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "annotated").mkdir(exist_ok=True)

    model = YOLO(model_path)
    img = Image.open(image_path).convert("RGB")
    img_area = img.width * img.height

    # Предсказание
    results = model.predict(image_path, conf=0.1, imgsz=640, save=False)
    r = results[0]

    # Словарь для подсчёта площадей каждого класса
    class_areas = {name: 0 for name in model.names.values()}

    # Создаём копию картинки для отрисовки боксов
    draw = ImageDraw.Draw(img)

    for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        area = (x2 - x1) * (y2 - y1)
        cls_name = model.names[int(cls_id)]
        class_areas[cls_name] = class_areas.get(cls_name, 0) + area

        # Рисуем бокс и название
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), cls_name, fill="red")

    # Перевод в проценты
    percentages = {
        cls: round((area / img_area) * 100, 2)
        for cls, area in class_areas.items()
    }

    # Вывод в консоль
    print(f"\n📊 Анализ изображения: {image_path}")
    for cls_name, percent in percentages.items():
        print(f"  🏷 {cls_name}: {percent:.2f}% площади изображения")
    print("\n✅ Анализ завершён")

    # Сохраняем результаты
    json_path = out_dir / "analysis.json"
    data = {
        "image": image_path,
        "results": percentages
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Сохраняем картинку
    annotated_path = out_dir / "annotated" / Path(image_path).name
    img.save(annotated_path)
    print(f"💾 Результаты сохранены в {json_path}")
    print(f"🖼️ Картинка сохранена в {annotated_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Путь к изображению")
    ap.add_argument("--model", required=True, help="Путь к модели YOLO (.pt)")
    args = ap.parse_args()

    analyze(args.image, args.model)
