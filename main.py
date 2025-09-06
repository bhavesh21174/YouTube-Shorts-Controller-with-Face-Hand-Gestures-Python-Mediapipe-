import cv2
import mediapipe as mp
import time
import pyautogui
import webbrowser
import warnings
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Volume Control Setup ----------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# ---------------- Open YouTube Shorts ----------------
youtube_url = "https://www.youtube.com/shorts/"
webbrowser.open(youtube_url)
time.sleep(5)

# Focus browser
screen_width, screen_height = pyautogui.size()
pyautogui.click(screen_width // 2, screen_height // 2)

# ---------------- Mediapipe Setup ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
last_action = time.time()

print("✅ Ready! Eyebrow → Next Short | Thumb + Index Gap → Volume Control")

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face landmarks
    face_results = face_mesh.process(rgb)

    # Hand landmarks
    hand_results = hands.process(rgb)

    # ---------- Eyebrow Detection ----------
    if face_results.multi_face_landmarks is not None:
        for landmarks in face_results.multi_face_landmarks:
            h, w, _ = frame.shape
            brow_y = int(landmarks.landmark[105].y * h)
            eye_y = int(landmarks.landmark[159].y * h)
            diff = eye_y - brow_y

            cv2.putText(frame, f"Diff: {diff}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if diff > 25 and time.time() - last_action > 2:
                print("➡ Eyebrow Up → Next Short")
                pyautogui.scroll(-500)
                last_action = time.time()

    # ---------- Hand Gesture for Volume (Thumb + Index) ----------
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Thumb tip (4) & Index tip (8)
            x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            # Distance
            distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            cv2.putText(frame, f"Thumb-Index Dist: {int(distance)}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Map distance → volume [20 - 200 px] → [0.0 - 1.0]
            min_dist, max_dist = 20, 200
            vol = np.interp(distance, [min_dist, max_dist], [0.0, 1.0])

            volume.SetMasterVolumeLevelScalar(vol, None)
            cv2.putText(frame, f"Volume: {int(vol*100)}%", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YouTube Shorts + Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
