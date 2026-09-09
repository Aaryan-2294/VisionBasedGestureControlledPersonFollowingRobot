import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_features import extract_features


MODEL_PATH = "gesture_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"


model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
training_features = model_data["training_features"]
training_labels = model_data["training_labels"]
distance_thresholds = model_data["distance_thresholds"]
K = model_data["distance_k"]


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

landmarker = vision.HandLandmarker.create_from_options(
    options
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open camera.")

    landmarker.close()

    exit()


print()
print("=" * 60)
print("LIVE GESTURE DISTANCE TEST")
print("=" * 60)
print()
print("Hold FOLLOW / STOP / DOCK and watch the distance.")
print()
print("Press Q or ESC to quit.")
print()


connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_hand(frame, hand):

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


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

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

        draw_hand(
            frame,
            hand
        )

        features = np.array(
            extract_features(hand),
            dtype=np.float32
        )

        probabilities = model.predict_proba(
            [features]
        )[0]

        prediction_index = probabilities.argmax()

        candidate = model.classes_[
            prediction_index
        ]

        confidence = probabilities[
            prediction_index
        ]

        class_samples = training_features[
            training_labels == candidate
        ]

        distances = np.linalg.norm(
            class_samples - features,
            axis=1
        )

        k = min(
            K,
            len(distances)
        )

        nearest_distances = np.sort(
            distances
        )[:k]

        distance = np.mean(
            nearest_distances
        )

        threshold = distance_thresholds[
            candidate
        ]

        accepted = (
            distance <= threshold
        )

        print(
            f"{candidate:6s} | "
            f"confidence={confidence:.3f} | "
            f"distance={distance:.4f} | "
            f"threshold={threshold:.4f} | "
            f"{'ACCEPT' if accepted else 'REJECT'}"
        )

        cv2.putText(
            frame,
            f"Prediction: {candidate}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.4f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Threshold: {threshold:.4f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "ACCEPTED" if accepted else "REJECTED",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Live Gesture Distance Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()

cv2.destroyAllWindows()

landmarker.close()

print()
print("Test stopped.")