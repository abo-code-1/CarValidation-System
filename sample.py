import json
from ultralytics import YOLO
from pathlib import Path

# Пути
MODEL_PATH = Path("models/scratch_and_dent/model.pt")
IMAGE_PATH = Path("dent.jpg")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Загрузка модели
model = YOLO(str(MODEL_PATH))

# Запуск предсказания
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.1,
    imgsz=640,
    save=True,
    project=str(RESULTS_DIR),
    name="predict_run",
    save_txt=True,
    save_conf=True
)

# Извлекаем данные в JSON
output = []
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls_id]
        output.append({
            "class": name,
            "confidence": round(conf * 100, 2)
        })

# Сохраняем как JSON
with open(RESULTS_DIR / "analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Анализ завершён. Результаты сохранены в results/analysis.json")
