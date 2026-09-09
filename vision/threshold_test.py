import os
import joblib
import numpy as np


MODEL_PATH = "gesture_model.pkl"

GESTURES = ["FOLLOW", "STOP", "DOCK"]


def calculate_knn_distances(features, k):
    """Calculate leave-one-out KNN distance for every sample."""

    distances = []

    for i in range(len(features)):

        sample = features[i]

        other_samples = np.delete(features, i, axis=0)

        all_distances = np.linalg.norm(
            other_samples - sample,
            axis=1
        )

        nearest = np.sort(all_distances)[:k]

        distances.append(np.mean(nearest))

    return np.array(distances)


def percentile_report(distances):

    percentiles = [50, 75, 90, 95, 97, 98, 99, 100]

    print()

    for percentile in percentiles:

        value = np.percentile(
            distances,
            percentile
        )

        print(
            f"{percentile:>3}th percentile: "
            f"{value:.4f}"
        )


def acceptance_report(distances, threshold):

    accepted = np.sum(distances <= threshold)

    total = len(distances)

    percentage = (
        accepted / total
    ) * 100.0

    print(
        f"Threshold {threshold:.4f}: "
        f"{accepted}/{total} accepted "
        f"({percentage:.2f}%)"
    )


def main():

    if not os.path.exists(MODEL_PATH):

        print(
            "ERROR: gesture_model.pkl not found."
        )

        return

    model_data = joblib.load(MODEL_PATH)

    training_features = np.asarray(
        model_data["training_features"],
        dtype=np.float32
    )

    training_labels = np.asarray(
        model_data["training_labels"]
    )

    K = model_data["distance_k"]

    stored_thresholds = (
        model_data["distance_thresholds"]
    )

    print()
    print("=" * 60)
    print("GESTURE DISTANCE THRESHOLD TEST")
    print("=" * 60)

    print()

    print(
        f"Total training samples: "
        f"{len(training_features)}"
    )

    print(
        f"Features per sample: "
        f"{training_features.shape[1]}"
    )

    print(
        f"K: {K}"
    )

    print()

    print("Stored thresholds:")

    for gesture in GESTURES:

        print(
            f"  {gesture}: "
            f"{stored_thresholds[gesture]:.4f}"
        )

    for gesture in GESTURES:

        print()
        print("=" * 60)
        print(f"{gesture}")
        print("=" * 60)

        class_features = training_features[
            training_labels == gesture
        ]

        print(
            f"Samples: "
            f"{len(class_features)}"
        )

        if len(class_features) <= K:

            print(
                "Not enough samples for "
                "leave-one-out KNN."
            )

            continue

        distances = calculate_knn_distances(
            class_features,
            K
        )

        print()

        print(
            f"Minimum distance: "
            f"{distances.min():.4f}"
        )

        print(
            f"Mean distance: "
            f"{distances.mean():.4f}"
        )

        print(
            f"Maximum distance: "
            f"{distances.max():.4f}"
        )

        print()

        print("Distance distribution:")

        percentile_report(distances)

        print()

        print("Acceptance using stored threshold:")

        acceptance_report(
            distances,
            stored_thresholds[gesture]
        )

        print()

        print("Acceptance at different thresholds:")

        for percentile in [90, 95, 97, 98, 99, 100]:

            threshold = np.percentile(
                distances,
                percentile
            )

            acceptance_report(
                distances,
                threshold
            )


if __name__ == "__main__":
    main()