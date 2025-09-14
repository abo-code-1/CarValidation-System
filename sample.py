# yolo detect predict \
#   model="models/scratch_and_dent/model.pt" \
#   source="link" \
#   conf=0.1 \
#   imgsz=640


import kagglehub

# Download latest version
path = kagglehub.dataset_download("anujms/car-damage-detection")

print("Path to dataset files:", path)