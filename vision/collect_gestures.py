import cv2
import csv
import time
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_features import extract_features


GESTURE_DATA_DIR = "gesture_data"
HAND_MODEL_PATH = "hand_landmarker.task"

GESTURE_DURATION = 10
SAMPLE_RATE = 20
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE

os.makedirs(GESTURE_DATA_DIR, exist_ok=True)

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    landmarker.close()
    exit()


gestures = ["FOLLOW", "STOP", "DOCK"]

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


for gesture in gestures:

    print()
    print("=" * 40)
    print(f"Prepare gesture: {gesture}")
    print("=" * 40)
    print("Recording starts in 2 seconds.")
    print("Keep the same gesture.")
    print("Vary position, distance and orientation.")
    print()

    countdown_start = time.time()

    while time.time() - countdown_start < 2:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        remaining = 2 - (time.time() - countdown_start)

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Starting in {remaining:.1f}s",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q or ESC to quit",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Gesture Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()


    samples = []

    start_time = time.time()
    last_sample_time = 0

    while time.time() - start_time < GESTURE_DURATION:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        current_time = time.time()
        elapsed = current_time - start_time
        remaining = max(0, GESTURE_DURATION - elapsed)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            for start, end in connections:

                x1 = int(
                    hand[start].x * frame.shape[1]
                )

                y1 = int(
                    hand[start].y * frame.shape[0]
                )

                x2 = int(
                    hand[end].x * frame.shape[1]
                )

                y2 = int(
                    hand[end].y * frame.shape[0]
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            if current_time - last_sample_time >= SAMPLE_INTERVAL:

                features = extract_features(hand)

                samples.append(features)

                last_sample_time = current_time

        cv2.putText(
            frame,
            f"Recording: {gesture}",
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
            "Move closer/farther, left/right and rotate",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Keep the same gesture",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {len(samples)}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Gesture Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()


    output_path = os.path.join(
        GESTURE_DATA_DIR,
        f"{gesture}.csv"
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [f"f{i}" for i in range(63)]
        )

        writer.writerows(samples)


    print(
        f"{gesture}: saved {len(samples)} samples"
    )


cap.release()
cv2.destroyAllWindows()
landmarker.close()

print()
print("=" * 40)
print("Gesture collection complete.")
print("=" * 40)