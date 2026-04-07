import cv2
import numpy as np
import pickle
import os
import face_recognition
from PIL import Image
from config import DATASET_PATH, ENCODING_PATH

# Load Haarcascade (face detector)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

data = {"encodings": [], "names": []}

if os.path.exists(ENCODING_PATH):
    with open(ENCODING_PATH, "rb") as f:
        data = pickle.load(f)


def detect_face(image):
    if image is None:
        return []
    if image.ndim == 2:
        gray = image
    else:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


def preprocess(face):
    try:
        if face is None:
            return None
        face = np.asarray(face)
        if face.size == 0:
            return None

        # Ensure uint8 and 2D/3D shape
        if face.dtype != np.uint8:
            # Clip to [0,255] then cast
            face = np.clip(face, 0, 255).astype(np.uint8)

        if face.ndim == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        elif face.ndim == 3 and face.shape[2] == 4:
            face = face[:, :, :3]
        elif face.ndim != 3 or face.shape[2] != 3:
            return None

        face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    except:
        return None


def _load_image_bgr_u8(path: str):
    """
    Robust loader that guarantees an 8-bit 3-channel BGR numpy image, or None.
    Handles odd JPEG variants (e.g., CMYK) via Pillow fallback.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
        return img

    try:
        pil = Image.open(path).convert("RGB")  # guarantees 8-bit RGB
        rgb = np.array(pil, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr, dtype=np.uint8)
    except Exception:
        return None


def encode_faces():
    known_encodings = []
    known_names = []

    for user in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user)
        if not os.path.isdir(user_path):
            continue

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)

            image = _load_image_bgr_u8(img_path)

            if image is None:
                continue

            faces = detect_face(image)

            if len(faces) == 0:
                print("No face:", img_path)
                continue

            (x, y, w, h) = faces[0]
            face = image[y:y+h, x:x+w]

            rgb = preprocess(face)

            if rgb is None:
                continue

            try:
                encodings = face_recognition.face_encodings(rgb)

                if len(encodings) == 0:
                    continue

                known_encodings.append(encodings[0])
                known_names.append(user)

            except Exception as e:
                print("Encoding Error:", e)
                continue

    new_data = {"encodings": known_encodings, "names": known_names}

    with open(ENCODING_PATH, "wb") as f:
        pickle.dump(new_data, f)

    global data
    data = new_data


def recognize_face(image):
    if not data["encodings"]:
        return "No Data", []

    faces = detect_face(image)

    if len(faces) == 0:
        return "No Face", []

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]

    rgb = preprocess(face)

    if rgb is None:
        return "No Face", []

    try:
        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) == 0:
            return "Unknown", []

        encoding = encodings[0]

        matches = face_recognition.compare_faces(
            data["encodings"], encoding, tolerance=0.5
        )

        if True in matches:
            idx = matches.index(True)
            return data["names"][idx], faces

        return "Unknown", faces

    except Exception as e:
        print("Recognition Error:", e)
        return "Error", []