import cv2
import mediapipe as mp
import pyautogui
import time
import pyttsx3
from ultralytics import YOLO

# ================ MODEL VE MODÜL YÜKLEME =================
yolo_model = YOLO('best.pt')
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# ================ AYARLAR =================
ACTION_COOLDOWN = 2.0
FINGER_UP_THRESHOLD = 0.12
screen_w, screen_h = pyautogui.size()
last_action_time = 0

# --- SES AYARI ---
engine = pyttsx3.init()
engine.setProperty('rate', 180)


def speak(text):
    engine.say(text)
    engine.runAndWait()


# =============== HELPERS ==================
def get_finger_status(lm):
    fingers = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lm[tip].y < lm[pip].y - FINGER_UP_THRESHOLD)
    thumb_up = lm[4].y < lm[3].y - 0.05
    return thumb_up, fingers


# ================ MAIN ====================
cap = cv2.VideoCapture(0)

# Çözünürlüğü Genişlet (Geniş Çalışma Alanı)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
        mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. ADIM: Yüzleri Tespit Et (Filtre için)
        face_boxes = []
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                fx, fy, fw, fh = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                face_boxes.append((fx, fy, fx + fw, fy + fh))
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Yüz kutusu

        # 2. ADIM: YOLO Nesne Tespiti ve Yüzdelik Skorlar
        detected_objects = []
        yolo_results = yolo_model(frame, stream=True, conf=0.15, device=0, verbose=False)

        for r in yolo_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = yolo_model.names[int(box.cls[0])]

                # Yüz Filtresi
                is_on_face = False
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                for (f_x1, f_y1, f_x2, f_y2) in face_boxes:
                    if f_x1 < center_x < f_x2 and f_y1 < center_y < f_y2:
                        is_on_face = True
                        break

                if is_on_face: continue

                detected_objects.append({"label": label, "box": (x1, y1, x2, y2), "conf": conf})

                # Nesne Kutusu ve Yüzdelik Gösterimi
                color = (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                conf_text = f"{label} %{int(conf * 100)}"
                cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 3. ADIM: El İşleme ve Komutlar
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                thumb_up, other_fingers = get_finger_status(lm)
                up_count = sum(other_fingers)
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)

                for obj in detected_objects:
                    ox1, oy1, ox2, oy2 = obj["box"]
                    if ox1 < ix < ox2 and oy1 < iy < oy2:
                        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 4)

                        if now - last_action_time > ACTION_COOLDOWN:
                            msg = ""
                            if obj["label"] == "cups" and thumb_up and up_count == 0:
                                pyautogui.press("playpause")
                                msg = "Müzik kontrol edildi"
                            elif obj["label"] == "phones" and up_count == 2:
                                pyautogui.hotkey('win', 'prtscr')
                                msg = "Ekran görüntüsü alındı"
                            elif obj["label"] == "pencils" and up_count >= 3:
                                pyautogui.press("volumemute")
                                msg = "Sesi kapatıldı"
                            elif obj["label"] == "keyboards" and not thumb_up and up_count == 0:
                                pyautogui.hotkey('win', 'd')
                                msg = "Masaüstüne dönüldü"

                            if msg:
                                speak(msg)
                                last_action_time = now

        # --- SOL ALT KÖŞE KONTROL PANELİ (SADELEŞTİRİLMİŞ) ---
        panel_y = h - 120
        # Arka plan gölgesi (Okunabilirlik için)
        cv2.rectangle(frame, (5, panel_y - 25), (320, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, "KONTROL REHBERI:", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        controls = [
            "Cup - Bas Parmak Yukari (Music Stop)",
            "Phone - 2 Parmak Acik (2) (SS)",
            "Keyboard - Yumruk (0) (Desktop)",
            "Pencil - Acik El (5) (Mute)"
        ]

        for i, text in enumerate(controls):
            cv2.putText(frame, text, (15, panel_y + 20 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Final Proje - Genis Aci & Yuz Filtreli", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
