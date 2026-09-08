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
POSE_MODEL_PATH = "pose_landmarker_full.task"
MODEL_PATH = "gesture_model.pkl"

GESTURE_DURATION = 10
SAMPLE_RATE = 20
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE

GESTURES = ["FOLLOW", "STOP", "DOCK"]

RECALIBRATION_HOLD_TIME = 3.0
RECALIBRATION_COUNTDOWN = 5.0
GESTURE_CONFIRMATION_TIME = 1.0

TARGET_LOCK_HOLD_TIME = 3.0
POSE_UPDATE_INTERVAL = 4
TARGET_VISIBILITY_THRESHOLD = 0.5

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

pose_base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL_PATH
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=4,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    landmarker.close()
    pose_landmarker.close()
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



def get_pose_center(pose, width, height):
    important_points = [11, 12, 23, 24]
    points = []

    for index in important_points:
        landmark = pose[index]

        if landmark.visibility > TARGET_VISIBILITY_THRESHOLD:
            points.append(
                (
                    int(landmark.x * width),
                    int(landmark.y * height)
                )
            )

    if not points:
        return None

    x = sum(point[0] for point in points) // len(points)
    y = sum(point[1] for point in points) // len(points)

    return x, y


def get_pose_box(pose, width, height):
    visible_points = []

    for landmark in pose:
        if landmark.visibility > TARGET_VISIBILITY_THRESHOLD:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:
                visible_points.append((x, y))

    if not visible_points:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]

    return min(xs), min(ys), max(xs), max(ys)


def associate_hand_with_pose(hand, poses, width, height):
    wrist = hand[0]
    hand_position = (
        int(wrist.x * width),
        int(wrist.y * height)
    )

    best_index = None
    best_distance = float("inf")

    for index, pose in enumerate(poses):
        center = get_pose_center(pose, width, height)
        box = get_pose_box(pose, width, height)

        if center is None or box is None:
            continue

        x1, y1, x2, y2 = box

        margin_x = max(120, int((x2 - x1) * 0.35))
        margin_y = max(160, int((y2 - y1) * 0.40))

        inside_region = (
            x1 - margin_x <= hand_position[0] <= x2 + margin_x
            and
            y1 - margin_y <= hand_position[1] <= y2 + margin_y
        )

        if not inside_region:
            continue

        current_distance = (
            (hand_position[0] - center[0]) ** 2 +
            (hand_position[1] - center[1]) ** 2
        ) ** 0.5

        if current_distance < best_distance:
            best_distance = current_distance
            best_index = index

    return best_index


def draw_target(frame, pose, width, height):
    box = get_pose_box(pose, width, height)

    if box is None:
        return

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2
    )

    center = get_pose_center(pose, width, height)

    if center is not None:
        cv2.circle(
            frame,
            center,
            5,
            (0, 255, 255),
            -1
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
print("Normal operation is active.")
print("FOLLOW / STOP / DOCK use the learned gestures.")
print("Hold CLOSED FIST for 3 seconds to lock target and recalibrate.")
print("Press Q or ESC to quit.")
print()


fist_start_time = None

candidate_gesture = None
gesture_start_time = None
confirmed_gesture = None

pose_timestamp = 0
frame_counter = 0
poses = []

target_locked = False
target_pose_index = None
target_center = None


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

    frame_counter += 1

    if frame_counter % POSE_UPDATE_INTERVAL == 0:

        pose_timestamp += 33

        pose_result = pose_landmarker.detect_for_video(
            mp_image,
            pose_timestamp
        )

        poses = pose_result.pose_landmarks

        if target_locked and target_center is not None and poses:

            closest_index = None
            closest_distance = float("inf")

            for index, pose in enumerate(poses):

                center = get_pose_center(
                    pose,
                    frame.shape[1],
                    frame.shape[0]
                )

                if center is None:
                    continue

                current_distance = (
                    (center[0] - target_center[0]) ** 2 +
                    (center[1] - target_center[1]) ** 2
                ) ** 0.5

                if current_distance < closest_distance:

                    closest_distance = current_distance
                    closest_index = index

            if closest_index is not None:

                target_pose_index = closest_index
                target_center = get_pose_center(
                    poses[target_pose_index],
                    frame.shape[1],
                    frame.shape[0]
                )

    prediction = "No hand"
    confidence = 0.0
    gesture_distance = 0.0
    threshold = 0.0

    fist_detected = False
    confirmation_progress = 0.0

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        draw_hand(frame, hand)

        fist_detected = is_closed_fist(hand)

        if not target_locked and fist_detected:

            hand_pose_index = (
                associate_hand_with_pose(
                    hand,
                    poses,
                    frame.shape[1],
                    frame.shape[0]
                )
                if poses else None
            )

            if hand_pose_index is not None:

                if fist_start_time is None:
                    fist_start_time = time.time()

                held_time = time.time() - fist_start_time

                progress = min(
                    held_time / TARGET_LOCK_HOLD_TIME,
                    1.0
                )

                candidate_gesture = None
                gesture_start_time = None
                confirmed_gesture = None

                put_text(
                    frame,
                    "TARGET ALLOCATION",
                    40,
                    0.8
                )

                put_text(
                    frame,
                    f"Hold: {held_time:.1f} / 3.0 sec",
                    80,
                    0.7
                )

                bar_x = 20
                bar_y = 110
                bar_width = 400
                bar_height = 30

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

                if held_time >= TARGET_LOCK_HOLD_TIME:

                    target_locked = True
                    target_pose_index = hand_pose_index
                    target_center = get_pose_center(
                        poses[target_pose_index],
                        frame.shape[1],
                        frame.shape[0]
                    )

                    fist_start_time = None

                    print()
                    print("=" * 50)
                    print("TARGET LOCKED")
                    print(f"Target person index: {target_pose_index}")
                    print("Starting gesture recalibration...")
                    print("=" * 50)
                    print()

                    success = run_calibration()

                    if not success:
                        break

                    model_data = load_model()

                    if model_data is None:
                        break

                    candidate_gesture = None
                    gesture_start_time = None
                    confirmation_progress = 0.0
                    confirmed_gesture = None

                    print()
                    print("=" * 50)
                    print("NEW MODEL LOADED")
                    print("Target remains locked.")
                    print("Returning to normal operation.")
                    print("=" * 50)
                    print()

            else:
                fist_start_time = None

        elif fist_detected:

            # A locked target's fist is not a new allocation trigger.
            fist_start_time = None

        else:

            fist_start_time = None

            # Keep the original gesture pipeline independent from pose
            # association so intermittent pose detection cannot reset
            # gesture recognition.
            prediction, confidence, gesture_distance, threshold = (
                predict_gesture(
                    hand,
                    model_data
                )
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
                            current_time -
                            gesture_start_time
                        )

                        if (
                            confirmation_progress
                            >= GESTURE_CONFIRMATION_TIME
                        ):

                            if confirmed_gesture != prediction:

                                confirmed_gesture = prediction

                                print()
                                print("=" * 40)
                                print(
                                    f"COMMAND CONFIRMED: {prediction}"
                                )
                                print("=" * 40)
                                print()

            else:

                candidate_gesture = None
                gesture_start_time = None
                confirmation_progress = 0.0
                confirmed_gesture = None

    else:

        fist_detected = False
        fist_start_time = None
        candidate_gesture = None
        gesture_start_time = None
        confirmation_progress = 0.0
        confirmed_gesture = None


    if target_locked and poses and target_pose_index is not None:

        if target_pose_index < len(poses):

            draw_target(
                frame,
                poses[target_pose_index],
                frame.shape[1],
                frame.shape[0]
            )

    if target_locked:

        put_text(
            frame,
            "TARGET LOCKED",
            375,
            0.7
        )

    if not fist_detected:

        put_text(
            frame,
            f"Prediction: {prediction}",
            40,
            0.8
        )

        put_text(
            frame,
            f"Confidence: {confidence:.2f}",
            75,
            0.7
        )

        put_text(
            frame,
            f"Distance: {gesture_distance:.3f}",
            110,
            0.7
        )

        if result.hand_landmarks:

            put_text(
                frame,
                f"Threshold: {threshold:.3f}",
                145,
                0.7
            )

        if candidate_gesture is not None:

            progress = min(
                confirmation_progress /
                GESTURE_CONFIRMATION_TIME,
                1.0
            )

            put_text(
                frame,
                f"Confirming: {candidate_gesture}",
                180,
                0.65
            )

            put_text(
                frame,
                f"Confirmation: {confirmation_progress:.1f} / 1.0 sec",
                215,
                0.65
            )

            bar_x = 20
            bar_y = 240
            bar_width = 400
            bar_height = 25

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

            put_text(
                frame,
                f"COMMAND: {confirmed_gesture}",
                300,
                0.75
            )

        put_text(
            frame,
            "Hold CLOSED FIST for 3 sec to recalibrate",
            340,
            0.55
        )


    cv2.imshow(
        "Gesture Controller",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()
pose_landmarker.close()

print()
print("Gesture controller stopped.")