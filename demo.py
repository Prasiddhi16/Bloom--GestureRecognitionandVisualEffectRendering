import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
import math
import time

model_path = "hand_landmarker.task"


class Flower:
    
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.root_x = random.randint(30, screen_w - 30)
        self.root_y = screen_h
        self.max_height = random.randint(120, 280)
        self.lean = random.uniform(-40, 40)

        self.growth_speed = random.uniform(5.0, 10.0)
        self.current_height = 0.0

        self.bloom_progress = 0.0
        self.bloom_speed = random.uniform(0.06, 0.12)
        self.bloomed = False

        self.num_petals = random.choice([5, 6, 7])
        self.petal_size = random.randint(10, 22)
        self.petal_rotation = random.uniform(0, 360)

        petal_palettes = [
            (255,  80, 160),
            (220,  80, 220),
            (255, 180, 220),
            (255, 220,  80),
            (255, 255, 255),
            (160,  80, 255),
            (255, 140,  80),
            (100, 220, 255),
        ]
        self.petal_color  = random.choice(petal_palettes)
        self.stem_color   = (50, 160, 60)
        self.leaf_color   = (60, 180, 50)
        self.center_color = (255, 220, 40)

        self.life = 1.0
        self.decay_delay = random.uniform(1.5, 3.0)
        self.decay_start = None
        self.decay_speed = random.uniform(0.012, 0.025)

        self.alive  = True
        self.paused = False

        # Fixed leaf sides so they don't flip every frame
        self.leaf_sides = [1 if random.random() > 0.5 else -1 for _ in range(2)]

    def tip(self):
        t = self.current_height / max(self.max_height, 1)
        tip_x = self.root_x + self.lean * t
        tip_y = self.root_y - self.current_height
        return int(tip_x), int(tip_y)

    def stem_point(self, t):
        ctrl_x = self.root_x + self.lean * 0.6
        ctrl_y = self.root_y - self.max_height * 0.55
        tip_x  = self.root_x + self.lean
        tip_y  = self.root_y - self.max_height
        x = (1-t)**2 * self.root_x + 2*(1-t)*t * ctrl_x + t**2 * tip_x
        y = (1-t)**2 * self.root_y + 2*(1-t)*t * ctrl_y + t**2 * tip_y
        return int(x), int(y)

    def update(self, now):
        if self.paused:
            return

        if self.current_height < self.max_height:
            self.current_height = min(self.current_height + self.growth_speed, self.max_height)

        if self.current_height >= self.max_height:
            if self.bloom_progress < 1.0:
                self.bloom_progress = min(self.bloom_progress + self.bloom_speed, 1.0)
            else:
                self.bloomed = True
                if self.decay_start is None:
                    self.decay_start = now

        if self.decay_start is not None:
            elapsed = now - self.decay_start
            if elapsed > self.decay_delay:
                self.life -= self.decay_speed
                if self.life <= 0:
                    self.alive = False

    def draw(self, frame):
        if not self.alive or self.life <= 0:
            return

        alpha = max(0.0, min(self.life, 1.0))
        overlay = frame.copy()
        stem_progress = self.current_height / max(self.max_height, 1)

        # Stem
        steps = 20
        pts = [self.stem_point((i / steps) * stem_progress) for i in range(steps + 1)]
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i+1], self.stem_color, 2)

        # Leaves
        for idx, leaf_t in enumerate([0.35, 0.62]):
            if stem_progress < leaf_t:
                continue
            lx, ly = self.stem_point(leaf_t)
            side = self.leaf_sides[idx]
            leaf_len  = int(18 * alpha)
            leaf_wide = int(7  * alpha)
            if leaf_len < 2:
                continue
            cv2.ellipse(overlay, (lx, ly), (leaf_len, leaf_wide),
                        -40 * side, 0, 360, self.leaf_color, -1)

        # Flower head
        if self.bloom_progress > 0:
            tip_x, tip_y = self.tip()
            s = int(self.petal_size * self.bloom_progress)
            if s >= 2:
                rad = math.radians(self.petal_rotation)
                for i in range(self.num_petals):
                    angle = rad + (2 * math.pi / self.num_petals) * i
                    px = int(tip_x + math.cos(angle) * s * 1.4)
                    py = int(tip_y + math.sin(angle) * s * 1.4)
                    cv2.ellipse(overlay, (px, py),
                                (max(1, s), max(1, s // 2)),
                                math.degrees(angle), 0, 360,
                                self.petal_color, -1)
                cv2.circle(overlay, (tip_x, tip_y),
                           max(1, int(s * 0.55)), self.center_color, -1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# ─── GLITTER SPARKLE ──────────────────────────────────────────────────────────

class GlitterSparkle:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 14)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - random.uniform(0, 5)
        self.vy_gravity = random.uniform(0.05, 0.2)
        self.life  = 1.0
        self.decay = random.uniform(0.04, 0.08)
        self.size  = random.randint(2, 5)
        self.blink = random.random()
        colors = [(255,215,0),(220,220,255),(100,255,255),(255,255,255),(255,180,50),(180,255,180)]
        self.color  = random.choice(colors)
        self.paused = False

    def update(self):
        if self.paused:
            return
        self.x    += self.vx
        self.y    += self.vy
        self.vy   += self.vy_gravity
        self.vx   *= 0.97
        self.life -= self.decay
        self.blink = (self.blink + 0.3) % 1.0

    def draw(self, frame):
        if self.life <= 0:
            return
        twinkle = 0.5 + 0.5 * math.sin(self.blink * math.pi * 2)
        alpha = max(0.0, self.life * twinkle)
        cx, cy = int(self.x), int(self.y)
        s = self.size
        overlay = frame.copy()
        cv2.line(overlay, (cx-s, cy),   (cx+s, cy),   self.color, 1)
        cv2.line(overlay, (cx, cy-s),   (cx, cy+s),   self.color, 1)
        d = max(1, int(s * 0.7))
        cv2.line(overlay, (cx-d, cy-d), (cx+d, cy+d), self.color, 1)
        cv2.line(overlay, (cx+d, cy-d), (cx-d, cy+d), self.color, 1)
        cv2.circle(overlay, (cx, cy), max(1, s//2), self.color, -1)
        cv2.addWeighted(overlay, alpha * 0.9, frame, 1 - alpha * 0.9, 0, frame)

    @property
    def alive(self):
        return self.life > 0


class WaveTrail:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vy = random.uniform(-2, -0.5)
        self.vx = random.uniform(-1, 1)
        self.life  = 1.0
        self.decay = random.uniform(0.06, 0.12)
        self.size  = random.randint(3, 8)
        colors = [(100,200,255),(50,150,255),(180,100,255),(255,255,255)]
        self.color  = random.choice(colors)
        self.paused = False

    def update(self):
        if self.paused:
            return
        self.x    += self.vx
        self.y    += self.vy
        self.life -= self.decay

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = max(0.0, self.life)
        overlay = frame.copy()
        cv2.circle(overlay, (int(self.x), int(self.y)), self.size, self.color, -1)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)

    @property
    def alive(self):
        return self.life > 0



def is_open_palm(landmarks):
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
    return all(landmarks[tip].y < landmarks[mcp].y for tip, mcp in pairs)

def is_fist(landmarks):
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return all(landmarks[tip].y > landmarks[pip].y for tip, pip in pairs)

def draw_magic_glow(frame, x, y, radius=80, color=(180, 80, 255)):
    overlay = np.zeros_like(frame)
    for r in range(radius, 0, -10):
        cv2.circle(overlay, (x, y), r, color, -1)
    cv2.addWeighted(overlay, 0.28, frame, 1.0, 0, frame)

def draw_hud(frame, gesture_text, paused):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.5, frame, 0.5, 0, frame)
    if gesture_text:
        cv2.putText(frame, gesture_text, (20, h - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 200, 255), 2)
    if paused:
        cv2.putText(frame, "PAUSED", (w - 135, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
    cv2.putText(frame, "Q to quit", (w - 120, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)



def main():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector     = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    ret, test_frame = cap.read()
    SCREEN_H, SCREEN_W = (test_frame.shape[:2] if ret else (480, 640))

    flowers  = []
    sparkles = []
    trails   = []

    palm_cooldown   = 0.0
    wave_cooldown   = 0.0
    flower_cooldown = 0.0
    PALM_CD   = 0.10
    WAVE_CD   = 0.05
    FLOWER_CD = 0.15   # new flower every 0.15 s while palm is open

    prev_x = None
    paused = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result   = detector.detect(mp_image)

        gesture_text = ""

        if result.hand_landmarks:
            landmarks     = result.hand_landmarks[0]
            wrist_x       = int(landmarks[0].x * w)
            wrist_y       = int(landmarks[0].y * h)
            palm_open     = is_open_palm(landmarks)
            fist_detected = is_fist(landmarks)

            
            if fist_detected:
                paused = True
                gesture_text = "Fist — animations paused"
            elif palm_open:
                paused = False

            if not paused:
                if palm_open:
                    gesture_text = " Open palm → Blooming!"
                    draw_magic_glow(frame, wrist_x, wrist_y)

                    # New flower from the bottom of the screen
                    if now - flower_cooldown > FLOWER_CD:
                        flower_cooldown = now
                        flowers.append(Flower(w, h))

                    # Glitter from fingertips
                    if now - palm_cooldown > PALM_CD:
                        palm_cooldown = now
                        for tip_idx in [4, 8, 12, 16, 20]:
                            tx = int(landmarks[tip_idx].x * w)
                            ty = int(landmarks[tip_idx].y * h)
                            for _ in range(3):
                                sparkles.append(GlitterSparkle(tx, ty))

                if prev_x is not None and abs(wrist_x - prev_x) > 40:
                    gesture_text = "Wave detected!"
                    if now - wave_cooldown > WAVE_CD:
                        wave_cooldown = now
                        for _ in range(12):
                            trails.append(WaveTrail(wrist_x, wrist_y))

            prev_x = wrist_x

        
        for p in flowers + sparkles + trails:
            p.paused = paused

        live = []
        for f in flowers:
            f.update(now)
            if f.alive:
                f.draw(frame)
                live.append(f)
        flowers = live

        live = []
        for s in sparkles:
            s.update()
            if s.alive:
                s.draw(frame)
                live.append(s)
        sparkles = live

       
        live = []
        for t in trails:
            t.update()
            if t.alive:
                t.draw(frame)
                live.append(t)
        trails = live

        draw_hud(frame, gesture_text, paused)
        cv2.imshow(" Gesture Magic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

