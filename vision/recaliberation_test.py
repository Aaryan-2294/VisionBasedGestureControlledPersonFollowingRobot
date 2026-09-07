import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_MODEL_PATH = "hand_landmarker.task"

CONFIRMATION_TIME = 3.0


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


def distance(a, b):
    return np.linalg.norm(
        np.array([a.x, a.y, a.z]) -
        np.array([b.x, b.y, b.z])
    )


def is_closed_fist(hand):

    wrist = hand[0]

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    folded_fingers = 0

    for fingertip, pip in finger_pairs:

        tip_distance = distance(
            hand[fingertip],
            wrist
        )

        pip_distance = distance(
            hand[pip],
            wrist
        )

        if tip_distance < pip_distance:
            folded_fingers += 1


    if folded_fingers != 4:
        return False


    thumb_tip = hand[4]
    index_mcp = hand[5]
    pinky_mcp = hand[17]

    thumb_to_index = distance(
        thumb_tip,
        index_mcp
    )

    thumb_to_pinky = distance(
        thumb_tip,
        pinky_mcp
    )

    palm_width = distance(
        index_mcp,
        pinky_mcp
    )


    thumb_folded = (
        thumb_to_index < palm_width * 0.9
        and
        thumb_to_pinky < palm_width * 0.9
    )


    return thumb_folded


fist_start_time = None
recalibration_triggered = False


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

    fist_detected = False


    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        fist_detected = is_closed_fist(hand)


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


    if fist_detected:

        if fist_start_time is None:

            fist_start_time = (
                cv2.getTickCount()
                / cv2.getTickFrequency()
            )

        current_time = (
            cv2.getTickCount()
            / cv2.getTickFrequency()
        )

        held_time = (
            current_time -
            fist_start_time
        )

        progress = min(
            held_time / CONFIRMATION_TIME,
            1.0
        )


        cv2.putText(
            frame,
            "RECALIBRATION GESTURE",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Hold: {held_time:.1f}s / 3.0s",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        bar_width = 300

        filled_width = int(
            bar_width * progress
        )


        cv2.rectangle(
            frame,
            (20, 105),
            (20 + bar_width, 130),
            (255, 255, 255),
            2
        )

        cv2.rectangle(
            frame,
            (20, 105),
            (20 + filled_width, 130),
            (0, 255, 255),
            -1
        )


        if held_time >= CONFIRMATION_TIME:

            recalibration_triggered = True

            cv2.putText(
                frame,
                "RECALIBRATION REQUESTED!",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


    else:

        fist_start_time = None

        if not recalibration_triggered:

            cv2.putText(
                frame,
                "Normal operation",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Hold closed fist for 3 seconds",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


    if recalibration_triggered:

        cv2.putText(
            frame,
            "RECALIBRATION REQUESTED!",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Release fist to reset",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


    cv2.imshow(
        "Recalibration Gesture Test",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()