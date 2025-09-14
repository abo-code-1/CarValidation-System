# src/detect.py
from ultralytics import YOLO
import argparse, json
from PIL import Image

def side_of_image(x_center_norm):
    # 0..1: <0.33=left, 0.33..0.66=center, >0.66=right
    if x_center_norm < 0.33: return "left"
    if x_center_norm > 0.66: return "right"
    return "center"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--save_vis", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.predict(args.image, conf=args.conf, save=args.save_vis)

    # Чтобы вычислить сторону — нужно знать ширину
    w, h = Image.open(args.image).size

    out = []
    r = results[0]
    for b in r.boxes:
        cls_id = int(b.cls)
        cls_name = r.names[cls_id]
        conf = float(b.conf)
        x1,y1,x2,y2 = map(float, b.xyxy[0])
        cx = (x1+x2)/2.0
        x_center_norm = cx / w
        out.append({
            "class": cls_name,
            "confidence": conf,
            "bbox_xyxy": [x1,y1,x2,y2],
            "side_in_image": side_of_image(x_center_norm)  # left/center/right в координатах ФОТО
        })

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
