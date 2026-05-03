import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the downloaded model file
model_path = "hand_landmarker.task"

# Configure options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

# Create detector
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_x = None

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Run detection
    result = detector.detect(mp_image)

    gesture_text = ""

    if result.hand_landmarks:
        # Get wrist x position (landmark 0)
        h, w, c = frame.shape
        wrist_x = int(result.hand_landmarks[0][0].x * w)

        # Detect wave (side-to-side motion)
        if prev_x is not None and abs(wrist_x - prev_x) > 40:
            gesture_text = "Wave detected → Ending magic ✨"
        prev_x = wrist_x

        # Detect open palm (simple check: index finger extended)
        index_tip = result.hand_landmarks[0][8].y
        index_mcp = result.hand_landmarks[0][5].y
        if index_tip < index_mcp:
            gesture_text = "Open palm detected → Blooming flowers 🌸"

    # Show text on screen
    cv2.putText(frame, gesture_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Gesture Demo (Tasks API)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()