import cv2 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
def is_pinch(landmarks):
    thumb_tip=landmarks.landmark[4]
    index_tip=landmarks.landmark[8]
    dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return dist < 0.05
cap=cv2.VideoCapture(0)

while cap.isOpened():
    sucess,frame=cap.read()
    if not sucess:
        break
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results   = detector.detect(mp_image)
    if results.hand_landmarks and len(results.hand_landmarks) == 2:
        hand1, hand2 = results.hand_landmarks

        h,w,_=frame.shape
        h1_index=(int(hand1.landmark[8].x*w),int(hand1.landmark[8].y * h))
        h1_thumb = (int(hand1.landmark[4].x * w), int(hand1.landmark[4].y * h))
        h2_index = (int(hand2.landmark[8].x * w), int(hand2.landmark[8].y * h))
        h2_thumb = (int(hand2.landmark[4].x * w), int(hand2.landmark[4].y * h))

        x_min1 = min(h1_index[0], h1_thumb[0])
        y_min1 = min(h1_index[1], h1_thumb[1])
        x_max1 = max(h1_index[0], h1_thumb[0])
        y_max1 = max(h1_index[1], h1_thumb[1])
        cv2.rectangle(frame, (x_min1, y_min1), (x_max1, y_max1), (255,0,0), 2) 

        x_min2 = min(h2_index[0], h2_thumb[0])
        y_min2 = min(h2_index[1], h2_thumb[1])
        x_max2 = max(h2_index[0], h2_thumb[0])
        y_max2 = max(h2_index[1], h2_thumb[1])
        cv2.rectangle(frame, (x_min2, y_min2), (x_max2, y_max2), (0,0,255), 2) 

        if is_pinch(hand1):
            filename = "Player1_puzzle.jpg"
            cv2.imwrite(filename, frame[y_min1:y_max1, x_min1:x_max1])
            print(f"Player 1 puzzle image saved as {filename}")

        if is_pinch(hand2):
            filename = "Player2_puzzle.jpg"
            cv2.imwrite(filename, frame[y_min2:y_max2, x_min2:x_max2])
            print(f"Player 2 puzzle image saved as {filename}")

    cv2.imshow("Puzzle Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()