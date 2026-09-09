import cv2
import numpy as np
import torch
import torchreid

from ultralytics import YOLO


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "botsort_reid.yaml"

# Cosine similarity threshold.
# We will tune this after testing.
REID_THRESHOLD = 0.65

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# ReID model
# ------------------------------------------------------------

print("Loading OSNet ReID model...")

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

reid_model.eval()
reid_model.to(DEVICE)

print(f"OSNet loaded on: {DEVICE}")


# ------------------------------------------------------------
# ReID preprocessing
# ------------------------------------------------------------

def prepare_person_crop(frame, box):
    """
    Crop a detected person and prepare it for OSNet.
    """

    x1, y1, x2, y2 = map(int, box)

    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # OSNet normally expects person images around 256x128.
    crop = cv2.resize(crop, (128, 256))

    # OpenCV: BGR
    # PyTorch/torchreid: RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    crop = crop.astype(np.float32) / 255.0

    # ImageNet normalization used by TorchReID models.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    crop = (crop - mean) / std

    # HWC -> CHW
    crop = np.transpose(crop, (2, 0, 1))

    crop = torch.from_numpy(
        crop.astype(np.float32)
    )

    # Add batch dimension.
    crop = crop.unsqueeze(0)

    return crop.to(DEVICE)


# ------------------------------------------------------------
# Extract ReID embedding
# ------------------------------------------------------------

@torch.no_grad()
def extract_embedding(frame, box):
    """
    Extract an OSNet appearance embedding from a person.
    """

    crop = prepare_person_crop(frame, box)

    if crop is None:
        return None

    embedding = reid_model(crop)

    # Normalize embedding.
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    embedding = embedding[0].cpu().numpy()

    return embedding


# ------------------------------------------------------------
# Compare two embeddings
# ------------------------------------------------------------

def cosine_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two ReID embeddings.
    """

    if embedding_a is None or embedding_b is None:
        return 0.0

    a = embedding_a / (
        np.linalg.norm(embedding_a) + 1e-8
    )

    b = embedding_b / (
        np.linalg.norm(embedding_b) + 1e-8
    )

    return float(np.dot(a, b))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    target_embedding = None
    target_track_id = None

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print()
    print("==============================================")
    print("        OSNet Persistent Person ReID Test")
    print("==============================================")
    print()
    print("Controls:")
    print("  T = select current person as target")
    print("  R = clear target")
    print("  Q / ESC = quit")
    print()
    print("Test sequence:")
    print("  1. Stand in front of the camera.")
    print("  2. Press T.")
    print("  3. Leave the frame.")
    print("  4. Return.")
    print("  5. Check whether TARGET is recovered.")
    print()
    print("Also test:")
    print("  - turning around")
    print("  - moving closer/farther")
    print("  - moving sideways")
    print("  - another person crossing")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera")
            break

        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # YOLO + BoT-SORT
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[0],
            verbose=False
        )

        result = results[0]

        detections = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Extract OSNet embeddings
            # ------------------------------------------------

            for box, track_id, confidence in zip(
                boxes,
                track_ids,
                confidences
            ):

                embedding = extract_embedding(
                    frame,
                    box
                )

                similarity = 0.0
                is_target = False

                if (
                    target_embedding is not None
                    and embedding is not None
                ):

                    similarity = cosine_similarity(
                        target_embedding,
                        embedding
                    )

                    if similarity >= REID_THRESHOLD:
                        is_target = True

                detections.append(
                    {
                        "box": box,
                        "track_id": track_id,
                        "confidence": confidence,
                        "embedding": embedding,
                        "similarity": similarity,
                        "is_target": is_target
                    }
                )

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        for detection in detections:

            x1, y1, x2, y2 = map(
                int,
                detection["box"]
            )

            track_id = detection["track_id"]

            confidence = detection["confidence"]

            similarity = detection["similarity"]

            is_target = detection["is_target"]

            if is_target:

                label = (
                    f"TARGET | ID {track_id} | "
                    f"ReID {similarity:.2f}"
                )

                thickness = 3

            else:

                label = (
                    f"Person | ID {track_id} | "
                    f"{confidence:.2f}"
                )

                thickness = 2

            # Green = target
            # White = other person

            box_color = (
                (0, 255, 0)
                if is_target
                else (255, 255, 255)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                thickness
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if target_embedding is None:

            status = "TARGET: NONE"

        else:

            status = (
                f"TARGET ENROLLED | "
                f"Original ID: {target_track_id}"
            )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "T=Select Target | R=Reset | Q=Quit",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "OSNet Persistent Person ReID",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Target enrollment
        # ----------------------------------------------------

        if key == ord("t"):

            if len(detections) == 0:

                print(
                    "No person detected. "
                    "Cannot select target."
                )

                continue

            # Select largest person.
            target = max(
                detections,
                key=lambda d:
                    (
                        (d["box"][2] - d["box"][0])
                        *
                        (d["box"][3] - d["box"][1])
                    )
            )

            if target["embedding"] is None:

                print(
                    "Could not extract "
                    "target ReID embedding."
                )

                continue

            target_embedding = (
                target["embedding"].copy()
            )

            target_track_id = target["track_id"]

            print()
            print("----------------------------------------------")
            print("TARGET ENROLLED")
            print(
                f"Original tracker ID: "
                f"{target_track_id}"
            )
            print(
                "OSNet appearance embedding saved."
            )
            print("----------------------------------------------")
            print()

        # ----------------------------------------------------
        # Clear target
        # ----------------------------------------------------

        elif key == ord("r"):

            target_embedding = None
            target_track_id = None

            print()
            print("Target cleared.")
            print()

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        elif key == ord("q") or key == 27:

            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()