import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the model file
model_path = "hand_landmarker.task"

# Create options for the hand landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


cap=cv2.VideoCapture(0)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (0,5), (5,6), (6,7), (7,8),       # Index
    (0,9), (9,10), (10,11), (11,12),  # Middle
    (0,13), (13,14), (14,15), (15,16),# Ring
    (0,17), (17,18), (18,19), (19,20) ]
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.line(frame, (50,50), (200,200), (255,0,0), 3) 
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),2)
    cv2.putText(frame, "Press Q to quit", (100,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    overlay = frame.copy()
    h,w,_=frame.shape
    cv2.rectangle(overlay, (0,0), (w,h), (255,0,0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, "Press Q to quit", (100,h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Run hand landmark detection
    result = detector.detect(mp_image)

    # Draw landmarks if detected
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            for lm in landmarks:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            for start_idx, end_idx in HAND_CONNECTIONS:
                        start = landmarks[start_idx]
                        end = landmarks[end_idx]
                        x1, y1 = int(start.x * frame.shape[1]), int(start.y * frame.shape[0])
                        x2, y2 = int(end.x * frame.shape[1]), int(end.y * frame.shape[0])
                        cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.imshow("Test",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()