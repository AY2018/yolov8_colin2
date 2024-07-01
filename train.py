from ultralytics import YOLO

model = YOLO("yolov8l-seg.yaml")  

results = model.train(data="dataset.yaml", epochs=10, imgsz=1280, batch=16)




