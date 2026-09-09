import os
import csv
import time
import math

import cv2
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_features import extract_features


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "gesture_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

GESTURES = [
    "FOLLOW",
    "STOP",
    "DOCK"
]

COLLECT_SECONDS = 8.0
WARMUP_SECONDS = 2.0

CAMERA_INDEX = 0

RESULTS_PATH = (
    "gesture_data/threshold_validation_results.csv"
)


# ============================================================
# Load model
# ============================================================

print("Loading gesture model...")

model_data = joblib.load(
    MODEL_PATH
)

model = model_data["model"]

training_features = model_data[
    "training_features"
]

training_labels = model_data[
    "training_labels"
]

distance_thresholds = model_data[
    "distance_thresholds"
]

distance_k = model_data[
    "distance_k"
]

print()
print("Stored thresholds:")

for gesture in GESTURES:

    print(
        f"  {gesture}: "
        f"{float(distance_thresholds[gesture]):.4f}"
    )

print(
    f"K = {distance_k}"
)

print()


# ============================================================
# MediaPipe
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# Distance calculation
# ============================================================

def mean_knn_distance(
    features,
    reference_features,
    k
):

    distances = []

    for reference in reference_features:

        distance = math.sqrt(
            sum(
                (
                    float(a) -
                    float(b)
                ) ** 2
                for a, b in zip(
                    features,
                    reference
                )
            )
        )

        distances.append(
            distance
        )

    distances.sort()

    k = min(
        k,
        len(distances)
    )

    return (
        sum(distances[:k]) /
        k
    )


def distance_to_gesture(
    features,
    gesture
):

    class_features = training_features[
        training_labels == gesture
    ]

    return mean_knn_distance(
        features,
        class_features,
        distance_k
    )


# ============================================================
# Statistics
# ============================================================

def percentile(
    values,
    p
):

    if not values:

        return 0.0

    values = sorted(
        values
    )

    index = (
        len(values) - 1
    ) * p

    lower = int(
        index
    )

    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        index - lower
    )

    return (
        values[lower]
        +
        (
            values[upper] -
            values[lower]
        )
        * fraction
    )


def print_statistics(
    gesture,
    records
):

    if not records:

        print(
            f"\n{gesture}: NO SAMPLES"
        )

        return

    distances = [
        record[
            "intended_distance"
        ]
        for record in records
    ]

    threshold = float(
        distance_thresholds[
            gesture
        ]
    )

    accepted = sum(
        distance <= threshold
        for distance in distances
    )

    predicted_correct = sum(
        record["predicted"] == gesture
        for record in records
    )

    print()
    print("=" * 60)
    print(gesture)
    print("=" * 60)

    print(
        f"Samples:       "
        f"{len(distances)}"
    )

    print(
        f"RF correct:    "
        f"{predicted_correct}/"
        f"{len(records)} "
        f"("
        f"{100.0 * predicted_correct / len(records):.2f}%"
        f")"
    )

    print()
    print(
        "Distance to INTENDED gesture class:"
    )

    print(
        f"  Minimum:     "
        f"{min(distances):.4f}"
    )

    print(
        f"  Mean:        "
        f"{sum(distances) / len(distances):.4f}"
    )

    print(
        f"  Median:      "
        f"{percentile(distances, 0.50):.4f}"
    )

    print(
        f"  90th:        "
        f"{percentile(distances, 0.90):.4f}"
    )

    print(
        f"  95th:        "
        f"{percentile(distances, 0.95):.4f}"
    )

    print(
        f"  98th:        "
        f"{percentile(distances, 0.98):.4f}"
    )

    print(
        f"  99th:        "
        f"{percentile(distances, 0.99):.4f}"
    )

    print(
        f"  Maximum:     "
        f"{max(distances):.4f}"
    )

    print()
    print(
        f"Current threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Current threshold acceptance: "
        f"{accepted}/{len(distances)} "
        f"("
        f"{100.0 * accepted / len(distances):.2f}%"
        f")"
    )

    print()


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        os.path.dirname(
            RESULTS_PATH
        ),
        exist_ok=True
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    print()
    print("=" * 60)
    print(
        "THRESHOLD VALIDATION TEST"
    )
    print("=" * 60)

    print()

    print(
        "This test DOES NOT retrain the model."
    )

    print(
        "This test DOES NOT modify "
        "gesture_model.pkl."
    )

    print()

    print("Controls:")
    print("  1 = FOLLOW")
    print("  2 = STOP")
    print("  3 = DOCK")
    print("  Q = Quit")

    print()

    print("For each gesture:")

    print(
        "  1. Press the corresponding number."
    )

    print(
        "  2. Wait through the warm-up."
    )

    print(
        "  3. Hold the gesture naturally."
    )

    print(
        "  4. Keep the gesture steady "
        "for the collection period."
    )

    print()

    print("IMPORTANT:")

    print(
        "Use the SAME gesture shapes "
        "you normally use."
    )

    print(
        "Do not deliberately hold the hand "
        "extremely rigid."
    )

    print()
    print("=" * 60)
    print()

    all_records = []

    timestamp_counter = 0

    current_gesture = None

    phase = "waiting"

    phase_start = 0.0

    gesture_records = {
        "FOLLOW": [],
        "STOP": [],
        "DOCK": []
    }

    recording_gesture = None

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        current_time = time.time()

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_counter += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_counter
        )

        predicted_gesture = None

        confidence = 0.0

        features = None

        intended_distance = None

        predicted_distance = None

        # ----------------------------------------------------
        # Hand detection
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand_landmarks = (
                result.hand_landmarks[0]
            )

            features = extract_features(
                hand_landmarks
            )

            probabilities = (
                model.predict_proba(
                    [features]
                )[0]
            )

            best_index = (
                probabilities.argmax()
            )

            predicted_gesture = (
                model.classes_[
                    best_index
                ]
            )

            confidence = float(
                probabilities[
                    best_index
                ]
            )

            # Distance to predicted class.
            predicted_distance = (
                distance_to_gesture(
                    features,
                    predicted_gesture
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Only calculate intended distance if a gesture
            # is actually selected.
            #
            # This prevents:
            #
            # distance_thresholds[None]
            #
            # ------------------------------------------------

            if (
                current_gesture is not None
                and
                current_gesture in GESTURES
            ):

                intended_distance = (
                    distance_to_gesture(
                        features,
                        current_gesture
                    )
                )

        # ----------------------------------------------------
        # Recording
        # ----------------------------------------------------

        if (
            phase == "recording"
            and
            recording_gesture is not None
        ):

            elapsed = (
                current_time -
                phase_start
            )

            if (
                features is not None
                and
                intended_distance is not None
            ):

                record = {

                    "gesture":
                        recording_gesture,

                    "time":
                        current_time,

                    "predicted":
                        predicted_gesture,

                    "confidence":
                        confidence,

                    "intended_distance":
                        intended_distance,

                    "predicted_distance":
                        predicted_distance
                }

                gesture_records[
                    recording_gesture
                ].append(
                    record
                )

                all_records.append(
                    record
                )

            if elapsed >= COLLECT_SECONDS:

                print_statistics(
                    recording_gesture,
                    gesture_records[
                        recording_gesture
                    ]
                )

                current_gesture = None

                recording_gesture = None

                phase = "waiting"

                print()

                print(
                    "Ready for next gesture:"
                )

                print(
                    "  1 = FOLLOW"
                )

                print(
                    "  2 = STOP"
                )

                print(
                    "  3 = DOCK"
                )

                print(
                    "  Q = Quit"
                )

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        elif (
            phase == "warmup"
            and
            current_gesture is not None
        ):

            elapsed = (
                current_time -
                phase_start
            )

            remaining = max(
                0.0,
                WARMUP_SECONDS -
                elapsed
            )

            if elapsed >= WARMUP_SECONDS:

                phase = "recording"

                recording_gesture = (
                    current_gesture
                )

                phase_start = (
                    current_time
                )

                print()

                print(
                    f"Recording "
                    f"{current_gesture}..."
                )

        # ----------------------------------------------------
        # Status overlay
        # ----------------------------------------------------

        if phase == "waiting":

            status = (
                "Press 1 / 2 / 3"
            )

        elif (
            phase == "warmup"
            and
            current_gesture is not None
        ):

            remaining = max(
                0.0,
                WARMUP_SECONDS -
                (
                    current_time -
                    phase_start
                )
            )

            status = (
                f"{current_gesture} "
                f"starting in "
                f"{remaining:.1f}s"
            )

        elif (
            phase == "recording"
            and
            recording_gesture is not None
        ):

            elapsed = (
                current_time -
                phase_start
            )

            remaining = max(
                0.0,
                COLLECT_SECONDS -
                elapsed
            )

            status = (
                f"RECORDING "
                f"{recording_gesture} "
                f"{remaining:.1f}s"
            )

        else:

            status = (
                "Press 1 / 2 / 3"
            )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Prediction overlay
        # ----------------------------------------------------

        if predicted_gesture is not None:

            prediction_text = (
                f"RF: "
                f"{predicted_gesture} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                prediction_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Distance overlay
        # ----------------------------------------------------

        if (
            intended_distance is not None
            and
            current_gesture is not None
            and
            current_gesture in GESTURES
        ):

            threshold = float(
                distance_thresholds[
                    current_gesture
                ]
            )

            distance_text = (
                f"{current_gesture} "
                f"distance: "
                f"{intended_distance:.4f} "
                f"/ threshold "
                f"{threshold:.4f}"
            )

            cv2.putText(
                frame,
                distance_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Camera window
        # ----------------------------------------------------

        cv2.imshow(
            "Threshold Validation",
            frame
        )

        key = cv2.waitKey(
            1
        ) & 0xFF

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if key == ord("q"):

            break

        # ----------------------------------------------------
        # Gesture selection
        # ----------------------------------------------------

        if phase == "waiting":

            # FOLLOW
            if key == ord("1"):

                current_gesture = (
                    "FOLLOW"
                )

                phase = "warmup"

                phase_start = (
                    current_time
                )

                print()
                print(
                    "FOLLOW selected."
                )

                print(
                    "Get into your normal "
                    "FOLLOW gesture."
                )

            # STOP
            elif key == ord("2"):

                current_gesture = (
                    "STOP"
                )

                phase = "warmup"

                phase_start = (
                    current_time
                )

                print()
                print(
                    "STOP selected."
                )

                print(
                    "Get into your normal "
                    "STOP gesture."
                )

            # DOCK
            elif key == ord("3"):

                current_gesture = (
                    "DOCK"
                )

                phase = "warmup"

                phase_start = (
                    current_time
                )

                print()
                print(
                    "DOCK selected."
                )

                print(
                    "Get into your normal "
                    "DOCK gesture."
                )

    # ========================================================
    # Cleanup
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    landmarker.close()

    # ========================================================
    # Save CSV
    # ========================================================

    if all_records:

        with open(
            RESULTS_PATH,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "gesture",
                    "time",
                    "predicted",
                    "confidence",
                    "intended_distance",
                    "predicted_distance"
                ]
            )

            writer.writeheader()

            writer.writerows(
                all_records
            )

        print()

        print(
            f"Results saved to: "
            f"{RESULTS_PATH}"
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()

    print("=" * 60)

    print(
        "FINAL VALIDATION SUMMARY"
    )

    print("=" * 60)

    for gesture in GESTURES:

        print_statistics(
            gesture,
            gesture_records[
                gesture
            ]
        )

    print()

    print(
        "Model was NOT modified."
    )

    print(
        "No retraining was performed."
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()