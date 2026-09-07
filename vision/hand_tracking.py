import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp += 1

        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape

                points = []

                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))

                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    start = points[connection.start]
                    end = points[connection.end]
                    cv2.line(frame, start, end, (0, 255, 0), 2)

                for point in points:
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)

        cv2.imshow("Robot Vision - MediaPipe Hand Tracking", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or key == 27:
            break

cap.release()
cv2.destroyAllWindows()