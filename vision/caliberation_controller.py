import cv2
import csv
import time
import os
import joblib
import numpy as np
import subprocess
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_features import extract_features

GESTURE_DATA_DIR = "gesture_data"
HAND_MODEL_PATH = "hand_landmarker.task"
MODEL_PATH = "gesture_model.pkl"

GESTURE_DURATION = 10
SAMPLE_RATE = 20
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE

GESTURES = ["FOLLOW", "STOP", "DOCK"]

RECALIBRATION_COUNTDOWN = 5.0
GESTURE_CONFIRMATION_TIME = 1.0

# Time a CLOSED FIST must be held (while IDLE) to lock in a new target.
PAIRING_HOLD_TIME = 3.0

# Max normalized-coordinate distance a hand's wrist may move between frames
# and still be considered the same tracked target. There is no true person
# re-identification here (MediaPipe only tracks a single hand at a time), so
# this is a positional-continuity heuristic. Tune it if the target is being
# "lost" too easily, or if a bystander's hand is too easily accepted.
MAX_TARGET_DISTANCE = 0.15

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


connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def is_closed_fist(hand):
    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    folded_fingers = 0

    for tip, pip in finger_pairs:
        if hand[tip].y > hand[pip].y:
            folded_fingers += 1

    index_mcp = hand[5]
    pinky_mcp = hand[17]
    thumb_tip = hand[4]

    palm_width = (
        ((index_mcp.x - pinky_mcp.x) ** 2 +
         (index_mcp.y - pinky_mcp.y) ** 2) ** 0.5
    )

    thumb_to_index = (
        ((thumb_tip.x - index_mcp.x) ** 2 +
         (thumb_tip.y - index_mcp.y) ** 2) ** 0.5
    )

    thumb_to_pinky = (
        ((thumb_tip.x - pinky_mcp.x) ** 2 +
         (thumb_tip.y - pinky_mcp.y) ** 2) ** 0.5
    )

    if palm_width < 1e-8:
        return False

    thumb_folded = (
        thumb_to_index < palm_width * 0.9 and
        thumb_to_pinky < palm_width * 0.9
    )

    return folded_fingers == 4 and thumb_folded


def draw_hand(frame, hand):
    for landmark in hand:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    for start, end in connections:
        x1 = int(hand[start].x * frame.shape[1])
        y1 = int(hand[start].y * frame.shape[0])

        x2 = int(hand[end].x * frame.shape[1])
        y2 = int(hand[end].y * frame.shape[0])

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


def put_text(frame, text, y, scale=0.7):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2
    )


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Gesture model not found.")
        print("Run train_gesture_model.py first.")
        return None

    return joblib.load(MODEL_PATH)


def predict_gesture(hand, model_data):
    model = model_data["model"]
    training_features = model_data["training_features"]
    training_labels = model_data["training_labels"]
    distance_thresholds = model_data["distance_thresholds"]
    K = model_data["distance_k"]

    features = np.array(
        extract_features(hand),
        dtype=np.float32
    )

    probabilities = model.predict_proba([features])[0]

    prediction_index = probabilities.argmax()

    candidate = model.classes_[prediction_index]

    confidence = probabilities[prediction_index]

    class_samples = training_features[
        training_labels == candidate
    ]

    distances = np.linalg.norm(
        class_samples - features,
        axis=1
    )

    k = min(K, len(distances))

    nearest_distances = np.sort(distances)[:k]

    distance = np.mean(nearest_distances)

    threshold = distance_thresholds[candidate]

    if distance <= threshold:
        prediction = candidate
    else:
        prediction = "REJECTED"

    return prediction, confidence, distance, threshold


def hand_position(hand):
    # Wrist landmark. Chosen over an all-landmark centroid because the wrist
    # stays relatively stable across different gestures (fingers move a lot,
    # the wrist doesn't), which makes it a steadier anchor for "is this the
    # same person's hand as before".
    wrist = hand[0]
    return (wrist.x, wrist.y)


def position_distance(p1, p2):
    return (
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    ) ** 0.5


def run_calibration():

    print()
    print("=" * 50)
    print("RECALIBRATION STARTED")
    print("=" * 50)
    print()

    countdown_start = time.time()

    while time.time() - countdown_start < RECALIBRATION_COUNTDOWN:

        ret, frame = cap.read()

        if not ret:
            return False

        frame = cv2.flip(frame, 1)

        remaining = (
            RECALIBRATION_COUNTDOWN -
            (time.time() - countdown_start)
        )

        put_text(
            frame,
            "RECALIBRATION STARTING",
            60,
            0.9
        )

        put_text(
            frame,
            f"Starting in {remaining:.1f}s",
            110,
            0.9
        )

        put_text(
            frame,
            "Prepare FOLLOW gesture",
            160,
            0.7
        )

        cv2.imshow(
            "Gesture Controller",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            return False

    for gesture in GESTURES:

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
                return False

            frame = cv2.flip(frame, 1)

            remaining = (
                2 -
                (time.time() - countdown_start)
            )

            put_text(
                frame,
                f"Gesture: {gesture}",
                45,
                0.9
            )

            put_text(
                frame,
                f"Starting in {remaining:.1f}s",
                85,
                0.8
            )

            put_text(
                frame,
                "Press Q or ESC to quit",
                120,
                0.6
            )

            cv2.imshow(
                "Gesture Controller",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                return False

        samples = []

        start_time = time.time()
        last_sample_time = 0

        while time.time() - start_time < GESTURE_DURATION:

            ret, frame = cap.read()

            if not ret:
                return False

            frame = cv2.flip(frame, 1)

            current_time = time.time()

            elapsed = current_time - start_time

            remaining = max(
                0,
                GESTURE_DURATION - elapsed
            )

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

                draw_hand(frame, hand)

                if (
                    current_time - last_sample_time
                    >= SAMPLE_INTERVAL
                ):

                    features = extract_features(hand)

                    samples.append(features)

                    last_sample_time = current_time

            put_text(
                frame,
                f"Recording: {gesture}",
                40,
                0.9
            )

            put_text(
                frame,
                f"Time remaining: {remaining:.1f}s",
                80,
                0.7
            )

            put_text(
                frame,
                "Move closer/farther, left/right and rotate",
                115,
                0.55
            )

            put_text(
                frame,
                "Keep the same gesture",
                145,
                0.6
            )

            put_text(
                frame,
                f"Samples: {len(samples)}",
                180,
                0.6
            )

            cv2.imshow(
                "Gesture Controller",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                return False

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

    print()
    print("=" * 50)
    print("Calibration complete")
    print("=" * 50)
    print()

    print("Training new gesture model...")

    training_result = subprocess.run(
        ["python", "train_gesture_model.py"],
        input="\n",
        text=True
    )

    if training_result.returncode != 0:

        print()
        print("=" * 50)
        print("MODEL TRAINING FAILED")
        print("=" * 50)
        print()

        return False

    print()
    print("=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)
    print()

    return True


model_data = load_model()

if model_data is None:
    landmarker.close()
    cap.release()
    exit()


print()
print("=" * 50)
print("Gesture Controller")
print("=" * 50)
print()
print("Robot starts IDLE (no target).")
print(f"Show a CLOSED FIST and hold it {PAIRING_HOLD_TIME:.0f}s to pair.")
print("Once paired, FOLLOW / STOP / DOCK gestures")
print("from the target are recognized; everyone")
print("else's gestures are ignored.")
print("Target gesturing DOCK relinquishes the")
print("target and returns the robot to IDLE.")
print("Press C to recalibrate (only while IDLE).")
print("Press Q or ESC to quit.")
print()


# "IDLE": no target paired, waiting for someone to pair via closed fist.
# "ACTIVE": a target is paired and being tracked/gestured to.
state = "IDLE"
target_position = None

pairing_fist_start_time = None

candidate_gesture = None
gesture_start_time = None
confirmed_gesture = None


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
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

    prediction = "No hand"
    confidence = 0.0
    distance = 0.0
    threshold = 0.0
    confirmation_progress = 0.0
    pairing_progress = 0.0
    is_target = False

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        draw_hand(frame, hand)

        current_position = hand_position(hand)
        fist_detected = is_closed_fist(hand)

        if state == "IDLE":

            candidate_gesture = None
            gesture_start_time = None
            confirmed_gesture = None

            if fist_detected:

                pairing_fist_start_time = (
                    pairing_fist_start_time
                    if pairing_fist_start_time is not None
                    else time.time()
                )

                held_time = time.time() - pairing_fist_start_time

                pairing_progress = min(
                    held_time / PAIRING_HOLD_TIME,
                    1.0
                )

                if held_time >= PAIRING_HOLD_TIME:

                    # Lock the new target first, then recalibrate all three
                    # learned gestures for that target. This is used both for
                    # the first target and after DOCK releases the previous
                    # target.
                    state = "ACTIVE"
                    target_position = current_position
                    pairing_fist_start_time = None

                    print()
                    print("=" * 40)
                    print("TARGET PAIRED")
                    print("Starting gesture recalibration...")
                    print("=" * 40)
                    print()

                    success = run_calibration()

                    if not success:
                        state = "IDLE"
                        target_position = None
                        candidate_gesture = None
                        gesture_start_time = None
                        confirmed_gesture = None
                        pairing_fist_start_time = None
                        break

                    model_data = load_model()

                    if model_data is None:
                        state = "IDLE"
                        target_position = None
                        candidate_gesture = None
                        gesture_start_time = None
                        confirmed_gesture = None
                        pairing_fist_start_time = None
                        break

                    candidate_gesture = None
                    gesture_start_time = None
                    confirmed_gesture = None

                    print()
                    print("=" * 50)
                    print("NEW GESTURE MODEL LOADED")
                    print("Returning to ACTIVE mode.")
                    print("=" * 50)
                    print()

            else:
                pairing_fist_start_time = None

        else:  # state == "ACTIVE"

            is_target = (
                position_distance(current_position, target_position)
                <= MAX_TARGET_DISTANCE
            )

            if is_target:

                target_position = current_position

                prediction, confidence, distance, threshold = (
                    predict_gesture(hand, model_data)
                )

                current_time = time.time()

                if prediction in GESTURES:

                    if prediction != candidate_gesture:

                        candidate_gesture = prediction
                        gesture_start_time = current_time

                        if confirmed_gesture != prediction:
                            confirmed_gesture = None

                    else:

                        if gesture_start_time is not None:

                            confirmation_progress = (
                                current_time - gesture_start_time
                            )

                            if (
                                confirmation_progress
                                >= GESTURE_CONFIRMATION_TIME
                            ):

                                if confirmed_gesture != prediction:

                                    confirmed_gesture = prediction

                                    print()
                                    print("=" * 40)
                                    print(f"COMMAND CONFIRMED: {prediction}")
                                    print("=" * 40)
                                    print()

                                    if prediction == "DOCK":

                                        print()
                                        print("=" * 50)
                                        print("DOCK confirmed.")
                                        print("Relinquishing target.")
                                        print("Returning to IDLE.")
                                        print("=" * 50)
                                        print()

                                        state = "IDLE"
                                        target_position = None
                                        candidate_gesture = None
                                        gesture_start_time = None
                                        confirmed_gesture = None
                                        pairing_fist_start_time = None

                else:
                    candidate_gesture = None
                    gesture_start_time = None
                    confirmed_gesture = None

            else:
                candidate_gesture = None
                gesture_start_time = None

    else:

        if state == "IDLE":
            pairing_fist_start_time = None
        else:
            candidate_gesture = None
            gesture_start_time = None


    # ---- overlay drawing ----

    if state == "IDLE":

        put_text(frame, "STATE: IDLE (no target)", 40, 0.8)

        if pairing_fist_start_time is not None:

            held_time = time.time() - pairing_fist_start_time

            put_text(
                frame,
                f"Pairing hold: {held_time:.1f} / {PAIRING_HOLD_TIME:.1f} sec",
                80,
                0.7
            )

            bar_x, bar_y, bar_width, bar_height = 20, 110, 400, 30

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2
            )

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + int(bar_width * pairing_progress),
                    bar_y + bar_height
                ),
                (0, 255, 0),
                -1
            )

        else:
            put_text(
                frame,
                "Show CLOSED FIST for 3s to pair",
                80,
                0.6
            )

        put_text(
            frame,
            "Press C to recalibrate (idle only)",
            115,
            0.55
        )

    else:

        if is_target:

            put_text(frame, "STATE: ACTIVE - target locked", 40, 0.8)
            put_text(frame, f"Prediction: {prediction}", 75, 0.7)
            put_text(frame, f"Confidence: {confidence:.2f}", 105, 0.6)
            put_text(frame, f"Distance: {distance:.3f}", 135, 0.6)
            put_text(frame, f"Threshold: {threshold:.3f}", 165, 0.6)

            if candidate_gesture is not None:

                progress = min(
                    confirmation_progress / GESTURE_CONFIRMATION_TIME,
                    1.0
                )

                put_text(
                    frame,
                    f"Confirming: {candidate_gesture}",
                    200,
                    0.6
                )

                bar_x, bar_y, bar_width, bar_height = 20, 220, 400, 25

                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (255, 255, 255),
                    2
                )

                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (
                        bar_x + int(bar_width * progress),
                        bar_y + bar_height
                    ),
                    (0, 255, 0),
                    -1
                )

            if confirmed_gesture is not None:
                put_text(frame, f"COMMAND: {confirmed_gesture}", 280, 0.75)

        elif result.hand_landmarks:

            put_text(frame, "STATE: ACTIVE - target locked", 40, 0.8)
            put_text(frame, "Hand detected: NOT the target", 75, 0.7)
            put_text(frame, "Gesture ignored", 105, 0.6)

        else:

            put_text(frame, "STATE: ACTIVE - target locked", 40, 0.8)
            put_text(frame, "Target not visible", 75, 0.7)

        put_text(
            frame,
            "Target gestures DOCK to release",
            320,
            0.55
        )


    cv2.imshow(
        "Gesture Controller",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    if key == ord("c") and state == "IDLE":

        success = run_calibration()

        if not success:
            break

        model_data = load_model()

        if model_data is None:
            break

        candidate_gesture = None
        gesture_start_time = None
        confirmed_gesture = None
        pairing_fist_start_time = None

        print()
        print("=" * 50)
        print("NEW MODEL LOADED")
        print("Returning to IDLE.")
        print("=" * 50)
        print()


cap.release()
cv2.destroyAllWindows()
landmarker.close()

print()
print("Gesture controller stopped.")
