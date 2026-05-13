import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

detector = vision.HandLandmarker.create_from_options(options)


def is_pinch(hand):
    thumb_tip = hand[4]
    index_tip = hand[8]

    dist = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )

    return dist < 0.05


cap = cv2.VideoCapture(0)

capture_count = 0
last_capture_time = 0
cooldown = 2  # seconds

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

   
    frame = cv2.resize(frame, (640, 480))

    h, w, _ = frame.shape

  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    results = detector.detect(mp_image)

    hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0

    # Debug: show number of hands detected
    cv2.putText(
        frame,
        f"Hands detected: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    if results.hand_landmarks:

      
        if len(results.hand_landmarks) >= 2:

            hand1 = results.hand_landmarks[0]
            hand2 = results.hand_landmarks[1]

            # Index finger tips
            p1 = (int(hand1[8].x * w), int(hand1[8].y * h))
            p2 = (int(hand2[8].x * w), int(hand2[8].y * h))

            # Rectangle coords
            x_min = min(p1[0], p2[0])
            y_min = min(p1[1], p2[1])
            x_max = max(p1[0], p2[0])
            y_max = max(p1[1], p2[1])

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                3
            )

            # Draw points
            cv2.circle(frame, p1, 10, (255, 0, 0), -1)
            cv2.circle(frame, p2, 10, (0, 0, 255), -1)

            cv2.putText(
                frame,
                "Bring hands apart + pinch BOTH to capture",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

           
            current_time = time.time()

            if (
                is_pinch(hand1)
                and is_pinch(hand2)
                and current_time - last_capture_time > cooldown
            ):

                cropped = frame[y_min:y_max, x_min:x_max]

                # Safety check
                if cropped.size > 0:

                    filename = f"capture_{capture_count}.jpg"
                    cv2.imwrite(filename, cropped)

                    print(f"Saved {filename}")

                    capture_count += 1
                    last_capture_time = current_time

                    cv2.putText(
                        frame,
                        "CAPTURED!",
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

    cv2.imshow("Gesture Rectangle Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()