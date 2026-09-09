import joblib
import numpy as np


MODEL_PATH = "gesture_model.pkl"

GESTURES = ["FOLLOW", "STOP", "DOCK"]

model_data = joblib.load(MODEL_PATH)

X = np.asarray(
    model_data["training_features"],
    dtype=np.float32
)

y = np.asarray(
    model_data["training_labels"]
)

K = model_data["distance_k"]


def calculate_distances(features, k):

    result = []

    for i in range(len(features)):

        sample = features[i]

        others = np.delete(
            features,
            i,
            axis=0
        )

        distances = np.linalg.norm(
            others - sample,
            axis=1
        )

        nearest = np.sort(distances)[:k]

        mean_distance = np.mean(nearest)

        result.append(mean_distance)

    return np.array(result)


print()
print("=" * 60)
print("GESTURE OUTLIER ANALYSIS")
print("=" * 60)
print()

print(f"K = {K}")

for gesture in GESTURES:

    print()
    print("=" * 60)
    print(gesture)
    print("=" * 60)

    mask = y == gesture

    class_features = X[mask]

    distances = calculate_distances(
        class_features,
        K
    )

    sorted_indices = np.argsort(
        distances
    )[::-1]

    print()
    print("Largest KNN distances:")
    print()

    for rank, index in enumerate(
        sorted_indices[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"distance = {distances[index]:.4f}"
        )

    print()

    print("Smallest KNN distances:")
    print()

    for rank, index in enumerate(
        sorted_indices[-5:][::-1],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"distance = {distances[index]:.4f}"
        )

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print()