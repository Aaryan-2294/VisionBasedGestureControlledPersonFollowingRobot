import os
import glob
import csv
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = "gesture_data"
MODEL_PATH = "gesture_model.pkl"

GESTURES = ["FOLLOW", "STOP", "DOCK"]


print()
print("=" * 40)
print("Gesture Model Training")
print("=" * 40)
print()


X = []
y = []


for gesture in GESTURES:

    file_path = os.path.join(
        DATA_DIR,
        f"{gesture}.csv"
    )

    if not os.path.exists(file_path):
        print(f"Missing dataset: {file_path}")
        exit()

    data = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=1
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    print(f"{gesture}: {len(data)} samples")

    X.extend(data.tolist())
    y.extend([gesture] * len(data))


X = np.array(X, dtype=np.float32)
y = np.array(y)


print()
print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print()


if os.path.exists(MODEL_PATH):

    print("WARNING:")
    print("An existing gesture_model.pkl was found.")
    print("Training will replace the existing model.")
    print()

    response = input(
        "Press ENTER to continue or type Q to cancel: "
    ).strip().lower()

    if response == "q":
        print("Training cancelled.")
        exit()


print()
print("=" * 40)
print("Preparing training and validation data")
print("=" * 40)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_test)}")


print()
print("=" * 40)
print("Training Random Forest")
print("=" * 40)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("=" * 40)
print("Training Results")
print("=" * 40)

print()
print(f"Accuracy: {accuracy:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


print()
print("=" * 40)
print("Calculating gesture distance thresholds")
print("=" * 40)


K = 5

distance_thresholds = {}


for gesture in GESTURES:

    train_samples = X_train[
        y_train == gesture
    ]

    validation_samples = X_test[
        y_test == gesture
    ]

    validation_distances = []

    for sample in validation_samples:

        distances = np.linalg.norm(
            train_samples - sample,
            axis=1
        )

        k = min(
            K,
            len(distances)
        )

        nearest_distances = np.sort(
            distances
        )[:k]

        mean_distance = np.mean(
            nearest_distances
        )

        validation_distances.append(
            mean_distance
        )


    threshold = np.percentile(
        validation_distances,
        95
    )

    distance_thresholds[gesture] = threshold

    print(
        f"{gesture}: threshold = {threshold:.4f}"
    )


model_data = {
    "model": model,
    "training_features": X_train,
    "training_labels": y_train,
    "distance_thresholds": distance_thresholds,
    "distance_k": K
}


joblib.dump(
    model_data,
    MODEL_PATH
)


print()
print("=" * 40)
print("Model saved successfully")
print("=" * 40)
print()
print(f"File: {MODEL_PATH}")
print()