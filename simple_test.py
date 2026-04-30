import numpy as np
import cv2
import mediapipe as mp
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, Image, ImageDraw
import tensorflow as tf

model = tf.keras.models.load_model('models/asl_model.h5', compile=False)

# MediaPipe Hand Landmarker (Tasks API)
latest_result = {"landmarks": None}

def result_callback(result, output_image, timestamp_ms):
    latest_result["landmarks"] = result.hand_landmarks[0] if result.hand_landmarks else None

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='models/hand_landmarker.task'),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )
)

def process_image(img):
    img = cv2.resize(img, (64, 64))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 64, 64, 3))
    img = img.astype('float32') / 255.
    return img

categories = [
    ["ain", 'ع'], ["al", "ال"], ["aleff", 'أ'], ["bb", 'ب'],
    ["dal", 'د'], ["dha", 'ط'], ["dhad", "ض"], ["fa", "ف"],
    ["gaaf", 'ج'], ["ghain", 'غ'], ["ha", 'ه'], ["haa", 'ه'],
    ["jeem", 'ج'], ["kaaf", 'ك'], ["khaa", 'خ'], ["la", 'لا'],
    ["laam", 'ل'], ["meem", 'م'], ["nun", "ن"], ["ra", 'ر'],
    ["saad", 'ص'], ["seen", 'س'], ["sheen", "ش"], ["ta", 'ت'],
    ["taa", 'ط'], ["thaa", "ث"], ["thal", "ذ"], ["toot", 'ت'],
    ["waw", 'و'], ["ya", "ى"], ["yaa", "ي"], ["zay", 'ز'],
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

cap = cv2.VideoCapture(0)
font = ImageFont.truetype("fonts/Sahel.ttf", 70)
smooth_box = None
frame_counter = 0
score, res, sequence = 0.0, '', ''
timestamp_ms = 0

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 33
    landmarker.detect_async(mp_image, timestamp_ms)

    hand_detected = False
    x1, y1, x2, y2 = 0, 0, 0, 0

    landmarks = latest_result["landmarks"]
    if landmarks is not None:
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        pad = int(max(x_max - x_min, y_max - y_min) * 0.35)
        half = max(x_max - x_min, y_max - y_min) // 2 + pad
        raw_box = (max(0, cx-half), max(0, cy-half), min(w, cx+half), min(h, cy+half))

        if smooth_box is None:
            smooth_box = raw_box
        else:
            smooth_box = tuple(int(s*0.65 + r*0.35) for s, r in zip(smooth_box, raw_box))

        x1, y1, x2, y2 = smooth_box
        hand_detected = True

        for (i, j) in HAND_CONNECTIONS:
            pt1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
            pt2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)
        for lm in landmarks:
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (255, 0, 255), -1)

    if hand_detected and (x2 - x1) > 20 and (y2 - y1) > 20:
        img_cropped = img[y1:y2, x1:x2]

        if frame_counter % 5 == 0 and img_cropped.size > 0:
            try:
                img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                proba = model.predict(process_image(img_rgb), verbose=0)[0]
                mx = np.argmax(proba)
                score = proba[mx] * 100
                res = categories[mx][0]
                sequence = categories[mx][1]
            except Exception:
                pass

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{res.upper()} ({score:.1f}%)", (x1, max(y1-15, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        smooth_box = None
        cv2.putText(img, "Show your hand to the camera",
                    (w//2 - 250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if sequence:
        reshaped_text = arabic_reshaper.reshape(sequence)
        bidi_text = get_display(reshaped_text)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, h - 90), bidi_text, (0, 255, 0), font=font)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    frame_counter += 1
    cv2.imshow("Arabic Sign Language", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
