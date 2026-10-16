from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 model (downloads automatically on first run)
model = YOLO("runs/detect/basketball_v1-2/weights/best.pt")
print(model.names)

# Open the video and grab a single frame partway through
image_path = "basketball-dataset/valid/images/5MRTCYEBVLX5_jpg.rf.e96efa04bf940b26a838ff9c50226e76.jpg"
frame = cv2.imread(image_path)


# Run detection on that single frame
results = model(frame)

    # Draw the detected boxes onto the frame and save it
annotated = results[0].plot()
cv2.imwrite("detection_test_output.jpg", annotated)
print("Saved detection_test_output.jpg — open it to see what was detected.")

    # Also print out what classes it found, in plain text
for box in results[0].boxes:
    class_name = model.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"Detected: {class_name} ({confidence:.2f} confidence)")
    