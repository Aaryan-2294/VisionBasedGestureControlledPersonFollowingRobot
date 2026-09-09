import os
import glob

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# Configuration
# ============================================================

GESTURE_DATA_DIR = "gesture_data"
MODEL_PATH = "gesture_model.pkl"

GESTURES = [
    "FOLLOW",
    "STOP",
    "DOCK"
]

K_NEIGHBORS = 5

VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

# Percentile used for the normal within-class variation.
BASE_PERCENTILE = 95

# Extra tolerance for live MediaPipe variation.
#
# The old thresholds were based almost entirely on very
# similar calibration frames and were therefore too strict.
#
# This multiplier deliberately gives the live gesture some
# room to vary while still keeping a finite rejection boundary.
LIVE_TOLERANCE_MULTIPLIER = 5.0

# Minimum useful threshold.
#
# This prevents a very tight calibration session from producing
# an unrealistically small rejection boundary.
MIN_DISTANCE_THRESHOLD = 0.20


# ============================================================
# Load gesture CSV files
# ============================================================

def load_gesture_data():

    all_features = []
    all_labels = []

    print()
    print("=" * 60)
    print("LOADING GESTURE DATA")
    print("=" * 60)

    for gesture in GESTURES:

        pattern = os.path.join(
            GESTURE_DATA_DIR,
            f"{gesture}.csv"
        )

        files = glob.glob(pattern)

        if not files:

            print(
                f"ERROR: No training file found for "
                f"{gesture}: {pattern}"
            )

            continue

        file_path = files[0]

        gesture_features = []

        with open(
            file_path,
            "r"
        ) as file:

            lines = file.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            values = line.split(",")

            try:

                features = [
                    float(value)
                    for value in values
                ]

            except ValueError:

                # Ignore header lines if present.
                continue

            if len(features) != 63:

                print(
                    f"WARNING: Ignoring invalid sample "
                    f"in {file_path}: "
                    f"{len(features)} features"
                )

                continue

            gesture_features.append(
                features
            )

        print(
            f"{gesture}: "
            f"{len(gesture_features)} samples"
        )

        for features in gesture_features:

            all_features.append(
                features
            )

            all_labels.append(
                gesture
            )

    X = np.asarray(
        all_features,
        dtype=np.float32
    )

    y = np.asarray(
        all_labels
    )

    return X, y


# ============================================================
# KNN distance
# ============================================================

def calculate_knn_distance(
    sample,
    reference_samples,
    k
):

    if len(reference_samples) == 0:

        return float("inf")

    differences = (
        reference_samples -
        sample
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    distances = np.sort(
        distances
    )

    k = min(
        k,
        len(distances)
    )

    return float(
        np.mean(
            distances[:k]
        )
    )


# ============================================================
# Calculate within-class distances
# ============================================================

def calculate_class_distances(
    X,
    y,
    gesture,
    k
):

    class_samples = X[
        y == gesture
    ]

    distances = []

    if len(class_samples) <= 1:

        return np.asarray(
            distances,
            dtype=np.float32
        )

    for i in range(
        len(class_samples)
    ):

        sample = class_samples[i]

        # Leave the current sample out so it cannot have
        # distance zero to itself.
        reference_samples = np.delete(
            class_samples,
            i,
            axis=0
        )

        distance = calculate_knn_distance(
            sample,
            reference_samples,
            k
        )

        distances.append(
            distance
        )

    return np.asarray(
        distances,
        dtype=np.float32
    )


# ============================================================
# Calculate personalized thresholds
# ============================================================

def calculate_thresholds(
    X,
    y
):

    print()
    print("=" * 60)
    print("CALCULATING PERSONALIZED DISTANCE THRESHOLDS")
    print("=" * 60)

    thresholds = {}

    for gesture in GESTURES:

        distances = calculate_class_distances(
            X,
            y,
            gesture,
            K_NEIGHBORS
        )

        if len(distances) == 0:

            thresholds[gesture] = (
                MIN_DISTANCE_THRESHOLD
            )

            continue

        base_threshold = float(
            np.percentile(
                distances,
                BASE_PERCENTILE
            )
        )

        mean_distance = float(
            np.mean(distances)
        )

        median_distance = float(
            np.median(distances)
        )

        maximum_distance = float(
            np.max(distances)
        )

        # ----------------------------------------------------
        # The old system used only the very tight calibration
        # distribution. That produced thresholds such as
        # 0.0619, which were too strict for live frames.
        #
        # We now expand the learned normal variation by a
        # controlled factor and enforce a minimum boundary.
        # ----------------------------------------------------

        expanded_threshold = (
            base_threshold *
            LIVE_TOLERANCE_MULTIPLIER
        )

        threshold = max(
            expanded_threshold,
            MIN_DISTANCE_THRESHOLD
        )

        thresholds[gesture] = (
            np.float32(threshold)
        )

        print()
        print(
            f"{gesture}:"
        )

        print(
            f"  Samples:              "
            f"{len(distances)}"
        )

        print(
            f"  Mean distance:        "
            f"{mean_distance:.4f}"
        )

        print(
            f"  Median distance:      "
            f"{median_distance:.4f}"
        )

        print(
            f"  {BASE_PERCENTILE}th percentile:    "
            f"{base_threshold:.4f}"
        )

        print(
            f"  Maximum distance:     "
            f"{maximum_distance:.4f}"
        )

        print(
            f"  Expanded threshold:   "
            f"{expanded_threshold:.4f}"
        )

        print(
            f"  Final threshold:      "
            f"{threshold:.4f}"
        )

    print()
    print("Final distance thresholds:")

    for gesture in GESTURES:

        print(
            f"  {gesture}: "
            f"{float(thresholds[gesture]):.4f}"
        )

    return thresholds


# ============================================================
# Train model
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PERSONALIZED GESTURE MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_gesture_data()

    if len(X) == 0:

        print()
        print(
            "ERROR: No valid training samples found."
        )

        return

    print()
    print(
        f"Total samples: "
        f"{len(X)}"
    )

    print(
        f"Features per sample: "
        f"{X.shape[1]}"
    )

    # --------------------------------------------------------
    # Verify gestures
    # --------------------------------------------------------

    print()
    print("Gesture sample counts:")

    for gesture in GESTURES:

        count = int(
            np.sum(
                y == gesture
            )
        )

        print(
            f"  {gesture}: {count}"
        )

        if count == 0:

            print()
            print(
                f"ERROR: No samples found for "
                f"{gesture}."
            )

            return

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print()
    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation)}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_validation,
            predictions,
            labels=GESTURES,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Distance thresholds
    #
    # IMPORTANT:
    #
    # Thresholds are calculated from ALL calibration samples,
    # not only the training split.
    #
    # This gives the personalized threshold calculation access
    # to the complete gesture calibration session.
    # --------------------------------------------------------

    thresholds = calculate_thresholds(
        X,
        y
    )

    # --------------------------------------------------------
    # Store model
    # --------------------------------------------------------

    model_data = {

        "model":
            model,

        # Keep ALL gesture features available for the
        # runtime KNN similarity calculation.
        "training_features":
            X,

        "training_labels":
            y,

        "distance_thresholds":
            thresholds,

        "distance_k":
            K_NEIGHBORS
    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Path: {MODEL_PATH}"
    )

    print()
    print(
        "Stored thresholds:"
    )

    for gesture in GESTURES:

        print(
            f"  {gesture}: "
            f"{float(thresholds[gesture]):.4f}"
        )

    print()
    print(
        "Training complete."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()