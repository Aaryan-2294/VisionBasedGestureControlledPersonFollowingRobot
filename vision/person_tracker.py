import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Person tracker test started.")
print("Press Q or ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
        break

    frame = cv2.flip(frame, 1)

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Person Tracker Test", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
