import cv2
import csv
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
DATA_DIR = os.path.join(os.path.dirname(__file__), "gesture_data")

RECORD_SECONDS = 10
SAMPLE_INTERVAL = 0.05

os.makedirs(DATA_DIR, exist_ok=True)

gesture = input("Enter gesture (FOLLOW / STOP / DOCK): ").strip().upper()

if gesture not in ["FOLLOW", "STOP", "DOCK"]:
    print("Invalid gesture.")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

data = []
start_time = None
last_sample_time = 0
frame_timestamp = 0

print(f"\nGet ready to record: {gesture}")
print("Recording will start automatically.")
print("Vary hand position, orientation and rotation.")
print("Keep the same gesture shape while moving naturally.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if start_time is None:
        start_time = time.time()

    elapsed = time.time() - start_time
    remaining = RECORD_SECONDS - elapsed

    if elapsed >= RECORD_SECONDS:
        break

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        for connection in connections:
            start = hand[connection.start]
            end = hand[connection.end]

            start_point = (
                int(start.x * frame.shape[1]),
                int(start.y * frame.shape[0])
            )

            end_point = (
                int(end.x * frame.shape[1]),
                int(end.y * frame.shape[0])
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (0, 255, 0),
                2
            )

        for landmark in hand:
            point = (
                int(landmark.x * frame.shape[1]),
                int(landmark.y * frame.shape[0])
            )

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )

        wrist = hand[0]

        features = []

        for landmark in hand:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])

        if elapsed - last_sample_time >= SAMPLE_INTERVAL:
            data.append(features)
            last_sample_time = elapsed

    cv2.putText(
        frame,
        f"Collecting: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Time remaining: {remaining:.1f}s",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Vary position, orientation & rotation",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Maintain the gesture shape",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.imshow("Gesture Data Collection", frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

filename = os.path.join(DATA_DIR, f"{gesture}.csv")

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("\nCollection complete.")
print(f"Gesture: {gesture}")
print(f"Samples collected: {len(data)}")
print(f"Saved to: {filename}")