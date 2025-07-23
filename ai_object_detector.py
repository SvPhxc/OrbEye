from ultralytics import YOLO
import cv2

# Load your model (custom or pretrained)
model = YOLO("yolov8n.pt")  # Replace with your custom model if needed

# Class filter (e.g., only detect 'person' which is class 0 in COCO)
TARGET_CLASSES = [0]  # Change this to your custom class ID(s)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# Run frame-by-frame tracking with filtering
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, persist=True, conf=0.3, tracker="bytetrack.yaml")

    annotated_frame = results[0].plot()

    # Filter results by class
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in TARGET_CLASSES:
            print(f"✅ Tracking object with class {cls_id} at {box.xywh[0].tolist()}")

    # Show annotated frame
    cv2.imshow("YOLOv8 Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
