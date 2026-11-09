from ultralytics import YOLO
import cv2
import sys

# Usage: python detect_test.py [model] [video]
# e.g.   python detect_test.py best.pt test_clip.mp4
model_path = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
video_path = sys.argv[2] if len(sys.argv) > 2 else "test_clip.mp4"
output_path = "detection_test_output.mp4"

# Load our trained model (best.pt is downloaded from Colab/Drive after training)
model = YOLO(model_path)
print("Classes:", model.names)

# Open the video and set up a writer for the annotated copy
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Count how many frames each class shows up in, and how confident the model was
frames = 0
seen = {name: 0 for name in model.names.values()}
confidences = {name: [] for name in model.names.values()}

while True:
    ok, frame = video.read()
    if not ok:
        break
    frames += 1

    # Run detection on this frame (conf=0.25 is YOLO's default cutoff)
    results = model(frame, conf=0.25, verbose=False)

    # Record the best detection of each class in this frame
    best = {}
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]
        best[name] = max(best.get(name, 0), float(box.conf[0]))
    for name, conf in best.items():
        seen[name] += 1
        confidences[name].append(conf)

    # Draw the boxes onto the frame and add it to the output video
    writer.write(results[0].plot())

video.release()
writer.release()

# Summary: what % of frames each object was found in
print(f"\nProcessed {frames} frames ({frames / fps:.1f}s)")
for name in model.names.values():
    if seen[name] == 0:
        continue
    avg_conf = sum(confidences[name]) / len(confidences[name])
    print(f"  {name:12s} found in {seen[name] / frames:6.1%} of frames, avg confidence {avg_conf:.2f}")
print(f"\nSaved {output_path} — open it to watch what the model saw.")
