import dlib
import cv2
import numpy as np
import math
from math import degrees
import os

# Load model dlib
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'shape_predictor_68_face_landmarks.dat')

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
except Exception as e:
    print(f"Gagal memuat model: {e}")
    exit()

def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detect_face_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None, "❌ Wajah tidak terdeteksi", image

    face = faces[0]
    shape = predictor(gray, face)
    points = np.array([(p.x, p.y) for p in shape.parts()])

    # Gambar landmark
    annotated = image.copy()
    for (x, y) in points:
        cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

    # Ukuran
    jaw_width = euclidean_distance(points[2], points[14])
    cheek_width = euclidean_distance(points[1], points[15])
    face_length = euclidean_distance(points[27], points[8])

    # Rasio
    length_cheek_ratio = face_length / cheek_width
    cheek_jaw_ratio = cheek_width / jaw_width

    # Debug info
    print(f"[DEBUG] Length/Cheek: {length_cheek_ratio:.2f}, Cheek/Jaw: {cheek_jaw_ratio:.2f}")

    # --- Final Logic ---
    if length_cheek_ratio >= 1.10:
        face_shape = "Oblong"
    elif length_cheek_ratio >= 0.95:
        if cheek_jaw_ratio >= 1.05:
            face_shape = "Oval"
        elif cheek_jaw_ratio < 0.95:
            face_shape = "Heart"
        else:
            face_shape = "Square"
    elif 0.85 <= length_cheek_ratio < 0.95:
        if cheek_jaw_ratio >= 1.05:
            face_shape = "Diamond"
        elif 0.95 <= cheek_jaw_ratio <= 1.05:
            face_shape = "Square"
        else:
            face_shape = "Round"
    else:  # < 0.85
        if cheek_jaw_ratio >= 1.05:
            face_shape = "Diamond"
        elif cheek_jaw_ratio < 0.95:
            face_shape = "Round"
        else:
            face_shape = "Round"
    
    return face_shape, None, annotated

def get_jaw_angle(landmarks):
    ax, ay = landmarks[3]
    bx, by = landmarks[4]
    cx, cy = landmarks[5]
    dx, dy = landmarks[6]

    alpha0 = math.atan2(cy - ay, cx - ax)
    alpha1 = math.atan2(dy - by, dx - bx)
    angle = abs(degrees(alpha1 - alpha0))
    return 180 - angle
