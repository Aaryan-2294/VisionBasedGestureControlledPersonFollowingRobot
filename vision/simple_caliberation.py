import cv2
import time
import threading
import urllib.request
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CAMERA_URL = "http://172.16.61.250:8080/video"
HAND_MODEL_PATH = "hand_landmarker.task"

FORWARD_SPEED = 0.20


# --------------------------------------------------
# LOW-LATENCY CAMERA
# --------------------------------------------------

class LowLatencyMJPEGCamera:

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.response = None

    def start(self):
        try:
            self.response = urllib.request.urlopen(
                self.url,
                timeout=5
            )
        except Exception as error:
            print(f"Could not open camera stream: {error}")
            return False

        self.running = True

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()

        return True

    def _reader(self):

        buffer = b""

        try:

            while self.running:

                chunk = self.response.read(4096)

                if not chunk:
                    break

                buffer += chunk

                while self.running:

                    start = buffer.find(b"\xff\xd8")

                    if start == -1:

                        if len(buffer) > 1:
                            buffer = buffer[-1:]

                        break

                    end = buffer.find(
                        b"\xff\xd9",
                        start + 2
                    )

                    if end == -1:

                        if start > 0:
                            buffer = buffer[start:]

                        break

                    jpg = buffer[start:end + 2]

                    buffer = buffer[end + 2:]

                    decoded = cv2.imdecode(
                        np.frombuffer(
                            jpg,
                            dtype=np.uint8
                        ),
                        cv2.IMREAD_COLOR
                    )

                    if decoded is not None:

                        with self.lock:
                            self.frame = decoded

        except Exception as error:

            if self.running:
                print(
                    f"Camera stream error: {error}"
                )

        finally:

            self.running = False

    def read(self):

        with self.lock:

            if self.frame is None:
                return False, None

            return True, self.frame.copy()

    def release(self):

        self.running = False

        try:

            if self.response is not None:
                self.response.close()

        except Exception:
            pass

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(timeout=1.0)


# --------------------------------------------------
# ROS 2 NODE
# --------------------------------------------------

class CalibrationController(Node):

    def __init__(self):

        super().__init__(
            "calibration_controller"
        )

        self.publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

    def send_command(
        self,
        linear,
        angular
    ):

        msg = Twist()

        msg.linear.x = linear
        msg.angular.z = angular

        self.publisher.publish(msg)

    def stop(self):

        self.send_command(
            0.0,
            0.0
        )


# --------------------------------------------------
# FINGER COUNT
# --------------------------------------------------

def count_fingers(hand):

    # MediaPipe landmark indices:
    #
    # Index:  8 tip, 6 PIP
    # Middle: 12 tip, 10 PIP
    # Ring:   16 tip, 14 PIP
    # Pinky:  20 tip, 18 PIP

    fingers = 0

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip, pip in finger_pairs:

        if hand[tip].y < hand[pip].y:
            fingers += 1

    return fingers


# --------------------------------------------------
# DRAW HAND
# --------------------------------------------------

connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]


def draw_hand(frame, hand):

    height, width = frame.shape[:2]

    for landmark in hand:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    for start, end in connections:

        x1 = int(
            hand[start].x * width
        )

        y1 = int(
            hand[start].y * height
        )

        x2 = int(
            hand[end].x * width
        )

        y2 = int(
            hand[end].y * height
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    rclpy.init()

    node = CalibrationController()

    camera = LowLatencyMJPEGCamera(
        CAMERA_URL
    )

    if not camera.start():

        node.destroy_node()
        rclpy.shutdown()

        return

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

    landmarker = vision.HandLandmarker.create_from_options(
        options
    )

    print()
    print("=" * 50)
    print("SIMPLE PHYSICAL ROVER TEST")
    print("=" * 50)
    print()
    print("1 FINGER  -> FORWARD")
    print("2 FINGERS -> STOP")
    print("NO HAND   -> STOP")
    print("Q / ESC   -> STOP + EXIT")
    print()
    print(f"Forward speed: {FORWARD_SPEED}")
    print()
    print("=" * 50)

    last_command = "STOP"

    try:

        while rclpy.ok():

            # Process ROS callbacks
            rclpy.spin_once(
                node,
                timeout_sec=0.0
            )

            ret, frame = camera.read()

            if not ret:

                node.stop()

                continue

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(
                mp_image
            )

            fingers = 0

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                draw_hand(
                    frame,
                    hand
                )

                fingers = count_fingers(
                    hand
                )

            # --------------------------------------
            # COMMAND LOGIC
            # --------------------------------------

            if fingers == 1:

                node.send_command(
                    FORWARD_SPEED,
                    0.0
                )

                command = "FORWARD"

            elif fingers == 2:

                node.stop()

                command = "STOP"

            else:

                node.stop()

                command = "STOP"

            # Print only when command changes
            if command != last_command:

                print(
                    f"Command: {command}"
                )

                last_command = command

            # --------------------------------------
            # DISPLAY
            # --------------------------------------

            cv2.putText(
                frame,
                f"Fingers: {fingers}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"COMMAND: {command}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "1 finger = FORWARD",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "2 fingers = STOP",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Physical Rover Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:

                break

    finally:

        # ALWAYS STOP THE ROBOT
        node.stop()

        time.sleep(0.2)

        camera.release()

        landmarker.close()

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()

        print()
        print("Physical rover test stopped.")


if __name__ == "__main__":
    main()