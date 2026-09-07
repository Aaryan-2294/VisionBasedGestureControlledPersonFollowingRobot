
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imshow("Robot Vision - Camera", frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
