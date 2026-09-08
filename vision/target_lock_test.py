import cv2
import time
import math
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"

MAX_POSES = 4
MAX_HANDS = 4

PAIRING_HOLD_TIME = 3.0
TRACK_DISTANCE_THRESHOLD = 180


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def is_fist(hand):
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

    index_mcp = np.array([
        hand[5].x,
        hand[5].y
    ])

    pinky_mcp = np.array([
        hand[17].x,
        hand[17].y
    ])

    thumb_tip = np.array([
        hand[4].x,
        hand[4].y
    ])

    palm_width = np.linalg.norm(
        index_mcp - pinky_mcp
    )

    if palm_width < 1e-6:
        return False

    thumb_to_index = np.linalg.norm(
        thumb_tip - index_mcp
    )

    thumb_to_pinky = np.linalg.norm(
        thumb_tip - pinky_mcp
    )

    return (
        folded_fingers == 4
        and thumb_to_index < palm_width * 0.9
        and thumb_to_pinky < palm_width * 0.9
    )


def get_pose_center(pose, width, height):
    important_points = [11, 12, 23, 24]

    points = []

    for index in important_points:
        landmark = pose[index]

        if landmark.visibility > 0.5:
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
        if landmark.visibility > 0.5:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:
                visible_points.append((x, y))

    if not visible_points:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )


def associate_hand_with_pose(hand, poses, width, height):
    wrist = hand[0]

    hand_position = (
        int(wrist.x * width),
        int(wrist.y * height)
    )

    best_pose = None
    best_distance = float("inf")

    for index, pose in enumerate(poses):
        pose_box = get_pose_box(
            pose,
            width,
            height
        )

        if pose_box is None:
            continue

        x1, y1, x2, y2 = pose_box

        expanded_x1 = x1 - 100
        expanded_y1 = y1 - 100
        expanded_x2 = x2 + 100
        expanded_y2 = y2 + 100

        if not (
            expanded_x1 <= hand_position[0] <= expanded_x2
            and
            expanded_y1 <= hand_position[1] <= expanded_y2
        ):
            continue

        center = get_pose_center(
            pose,
            width,
            height
        )

        if center is None:
            continue

        current_distance = distance(
            hand_position,
            center
        )

        if current_distance < best_distance:
            best_distance = current_distance
            best_pose = index

    return best_pose


def main():
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=POSE_MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=HAND_MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose_detector = vision.PoseLandmarker.create_from_options(
        pose_options
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        hand_options
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")

        pose_detector.close()
        hand_detector.close()

        return

    target_center = None
    target_locked = False

    fist_start_time = None
    target_pose_index = None

    frame_timestamp = 0

    print()
    print("======================================")
    print("          TARGET LOCK TEST")
    print("======================================")
    print()
    print("Initial state: NO TARGET")
    print()
    print("Hold a closed fist for 3 seconds.")
    print("The person performing the fist")
    print("will become the locked target.")
    print()
    print("Press Q to quit.")
    print("======================================")
    print()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_timestamp += 33

        pose_result = pose_detector.detect_for_video(
            mp_image,
            frame_timestamp
        )

        hand_result = hand_detector.detect_for_video(
            mp_image,
            frame_timestamp
        )

        poses = pose_result.pose_landmarks
        hands = hand_result.hand_landmarks

        pose_centers = []

        for pose in poses:
            center = get_pose_center(
                pose,
                width,
                height
            )

            pose_centers.append(center)

        fist_pose_index = None

        for hand in hands:

            if not is_fist(hand):
                continue

            fist_pose_index = associate_hand_with_pose(
                hand,
                poses,
                width,
                height
            )

            wrist = hand[0]

            wrist_x = int(wrist.x * width)
            wrist_y = int(wrist.y * height)

            cv2.circle(
                frame,
                (wrist_x, wrist_y),
                15,
                (255, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "FIST",
                (wrist_x + 15, wrist_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            break

        if not target_locked:

            if fist_pose_index is not None:

                if fist_start_time is None:
                    fist_start_time = time.time()

                elapsed = time.time() - fist_start_time

                remaining = max(
                    0,
                    PAIRING_HOLD_TIME - elapsed
                )

                cv2.putText(
                    frame,
                    f"PAIRING: {remaining:.1f}s",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )

                if elapsed >= PAIRING_HOLD_TIME:

                    if fist_pose_index < len(pose_centers):

                        candidate_center = pose_centers[
                            fist_pose_index
                        ]

                        if candidate_center is not None:

                            target_center = candidate_center
                            target_pose_index = fist_pose_index
                            target_locked = True

                            print()
                            print("TARGET LOCKED")
                            print(
                                f"Person index: "
                                f"{fist_pose_index}"
                            )
                            print()

                    fist_start_time = None

            else:

                fist_start_time = None

                cv2.putText(
                    frame,
                    "WAITING FOR FIST",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )

        else:

            closest_pose = None
            closest_distance = float("inf")

            for index, center in enumerate(pose_centers):

                if center is None:
                    continue

                if target_center is None:
                    continue

                current_distance = distance(
                    center,
                    target_center
                )

                if current_distance < closest_distance:
                    closest_distance = current_distance
                    closest_pose = index

            if (
                closest_pose is not None
                and
                closest_distance <= TRACK_DISTANCE_THRESHOLD
            ):

                target_pose_index = closest_pose

                target_center = pose_centers[
                    closest_pose
                ]

            else:

                target_pose_index = None

            if target_pose_index is not None:

                target_box = get_pose_box(
                    poses[target_pose_index],
                    width,
                    height
                )

                if target_box is not None:

                    x1, y1, x2, y2 = target_box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        frame,
                        "LOCKED TARGET",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            else:

                cv2.putText(
                    frame,
                    "TARGET LOST",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

        for index, pose in enumerate(poses):

            box = get_pose_box(
                pose,
                width,
                height
            )

            if box is None:
                continue

            if (
                target_locked
                and
                index == target_pose_index
            ):
                continue

            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"OTHER PERSON {index}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame,
            f"People detected: {len(poses)}",
            (30, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if target_locked:
            status = "TARGET LOCKED"
        else:
            status = "NO TARGET"

        cv2.putText(
            frame,
            status,
            (30, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Target Lock Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    pose_detector.close()
    hand_detector.close()


if __name__ == "__main__":
    main()