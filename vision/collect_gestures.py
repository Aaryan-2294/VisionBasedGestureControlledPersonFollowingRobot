import cv2
import mediapipe as mp
import csv
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

gestures = {
    "1": "FOLLOW",
    "2": "STOP",
    "3": "DOCK"
}

os.makedirs("gesture_data", exist_ok=True)

print("Select a gesture:")
print("1 - FOLLOW")
print("2 - STOP")
print("3 - DOCK")

choice = input("Enter choice: ")

if choice not in gestures:
    print("Invalid choice")
    exit()

label = gestures[choice]
filename = f"gesture_data/{label}.csv"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp = 0

    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        print(f"\nCollecting samples for: {label}")
        print("Hold your gesture in front of the camera.")
        print("Press Q to stop collecting.")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp += 1

            results = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]

                features = []

                wrist_x = landmarks[0].x
                wrist_y = landmarks[0].y

                for landmark in landmarks:
                    features.append(landmark.x - wrist_x)
                    features.append(landmark.y - wrist_y)
                    features.append(landmark.z)

                writer.writerow(features)

                cv2.putText(
                    frame,
                    f"Collecting: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Samples are being recorded",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                mp_landmarks = landmarks

                h, w, _ = frame.shape
                points = []

                for landmark in mp_landmarks:
                    points.append(
                        (int(landmark.x * w), int(landmark.y * h))
                    )

                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    start = points[connection.start]
                    end = points[connection.end]
                    cv2.line(frame, start, end, (0, 255, 0), 2)

                for point in points:
                    cv2.circle(frame, point, 4, (0, 0, 255), -1)

            cv2.imshow("Gesture Data Collection", frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord("q") or key == 27:
                break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinished collecting {label} samples.")
print(f"Saved to: {filename}")