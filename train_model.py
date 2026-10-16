from ultralytics import YOLO

# Start from the same pretrained weights we tested earlier —
# fine-tuning builds on top of this, rather than starting from zero
model = YOLO("yolov8m.pt")

# Train (fine-tune) it on our basketball dataset
results = model.train(
    data="Basketball and rim.v2i.yolov8/data.yaml",  # tells YOLO where images/labels/classes are
    epochs=200,           # how many full passes through the training data
    imgsz=640,            # resize all images to 640x640 before training (standard YOLO size)
    batch=4,              # how many images to process at once (lower = less memory used)
    name="basketball_V1", # names this training run, for organizing results later
    device = "mps",
    patience = 25         # faster than CPU run
    )