import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2,
min_hand_detection_confidence=0.3,
min_hand_presence_confidence=0.3,
 min_tracking_confidence=0.3)
detector = vision.HandLandmarker.create_from_options(options)

def finger_position(hand, w, h):
    return int(hand[8].x * w), int(hand[8].y * h)

def create_puzzle(image, grid_size=2):
    h, w, _ = image.shape
    tile_h, tile_w = h // grid_size, w // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            tile = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    random.shuffle(tiles)
    return tiles, tile_h, tile_w

def draw_puzzle(tiles, tile_h, tile_w, grid_size=2):
    puzzle_img = np.zeros((tile_h*grid_size, tile_w*grid_size, 3), dtype=np.uint8)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            puzzle_img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tiles[idx]
            idx += 1
    return puzzle_img

def is_pinch(hand):
    thumb_tip = hand[4]
    index_tip = hand[8]
    dist = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return dist < 0.05

def swap_tiles(tiles, idx1, idx2):
    tiles[idx1], tiles[idx2] = tiles[idx2], tiles[idx1]

def is_solved(tiles, original_tiles):
    return all((t == o).all() for t, o in zip(tiles, original_tiles))

cap = cv2.VideoCapture(0)
capture_count = 0
last_capture_time = 0
cooldown = 2
puzzle_mode = False
tiles, tile_h, tile_w, original_tiles = None, None, None, None
swipe_start = None
swipe_threshold = 50

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
    cv2.putText(frame, f"Hands detected: {hand_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if not puzzle_mode:
        if results.hand_landmarks and len(results.hand_landmarks) >= 2:
            hand1 = results.hand_landmarks[0]
            hand2 = results.hand_landmarks[1]
            p1 = (int(hand1[8].x * w), int(hand1[8].y * h))
            p2 = (int(hand2[8].x * w), int(hand2[8].y * h))
            x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
            x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.circle(frame, p1, 10, (255, 0, 0), -1)
            cv2.circle(frame, p2, 10, (0, 0, 255), -1)
            cv2.putText(frame, "Bring hands apart + pinch BOTH to capture",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            current_time = time.time()
            if (is_pinch(hand1) and is_pinch(hand2) and
                current_time - last_capture_time > cooldown):
                cropped = frame[y_min:y_max, x_min:x_max]
                if cropped.size > 0:
                    filename = f"capture_{capture_count}.jpg"
                    cv2.imwrite(filename, cropped)
                    capture_count += 1
                    last_capture_time = current_time
                    cv2.putText(frame, "CAPTURED!", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    tiles, tile_h, tile_w = create_puzzle(cropped, grid_size=2)
                    original_tiles, _, _ = create_puzzle(cropped, grid_size=2)
                    puzzle_mode = True
    else:
        puzzle_img = draw_puzzle(tiles, tile_h, tile_w, grid_size=2)
        cv2.putText(puzzle_img, "Pinch + swipe to swap tiles", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        puzzle_img = cv2.resize(puzzle_img, (w, h))
        alpha = 0.7
        frame = cv2.addWeighted(puzzle_img, alpha, frame, 1 - alpha, 0)
        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            if is_pinch(hand):
                grid_size = 2
                tip_x, tip_y = finger_position(hand, w, h)
                tip_x, tip_y = finger_position(hand, w, h)
                tile_row = tip_y // tile_h
                tile_col = tip_x // tile_w
                selected_idx = tile_row * grid_size + tile_col

                # Highlight selected tile
                x1, y1 = tile_col * tile_w, tile_row * tile_h
                x2, y2 = x1 + tile_w, y1 + tile_h
                cv2.rectangle(puzzle_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if swipe_start is None:
                    swipe_start = (tip_x, tip_y)
                else:
                    dx, dy = tip_x - swipe_start[0], tip_y - swipe_start[1]
                    if abs(dx) > swipe_threshold or abs(dy) > swipe_threshold:
                        idx1, idx2 = 0, 1 if abs(dx) > abs(dy) else 2
                        swap_tiles(tiles, idx1, idx2)
                        swipe_start = None
            else:
                swipe_start = None
        if is_solved(tiles, original_tiles):
            cv2.putText(frame, "Puzzle Solved!", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow("Gesture Rectangle Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
