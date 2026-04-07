import cv2
import numpy as np
import pickle
import os
import face_recognition
from PIL import Image
from config import DATASET_PATH, ENCODING_PATH

# Stricter matching to reduce false positives on unseen faces.
MATCH_TOLERANCE = 0.52
MIN_DISTANCE_GAP = 0.04

data = {"encodings": [], "names": []}

if os.path.exists(ENCODING_PATH):
    with open(ENCODING_PATH, "rb") as f:
        data = pickle.load(f)


def _to_bgr_u8(image):
    if image is None:
        return None
    image = np.asarray(image)
    if image.size == 0:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        return None
    return np.ascontiguousarray(image, dtype=np.uint8)


def _enhance_for_detection(image_bgr):
    # Improves detection for low light / slight side-angle shots.
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    enhanced = cv2.merge([y, cr, cb])
    return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)


def _detect_face_locations(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    locations = face_recognition.face_locations(rgb, model="hog")
    if locations:
        return rgb, locations

    enhanced_bgr = _enhance_for_detection(image_bgr)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    enhanced_rgb = np.ascontiguousarray(enhanced_rgb, dtype=np.uint8)
    locations = face_recognition.face_locations(enhanced_rgb, model="hog")
    return enhanced_rgb, locations


def _largest_face(locations):
    if not locations:
        return None
    return max(locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


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
            image = _to_bgr_u8(image)

            if image is None:
                continue

            try:
                rgb, locations = _detect_face_locations(image)
            except Exception as e:
                print("Encoding Error:", e)
                continue

            if len(locations) == 0:
                print("No face:", img_path)
                continue

            try:
                target_loc = _largest_face(locations)
                encodings = face_recognition.face_encodings(rgb, known_face_locations=[target_loc], num_jitters=2)

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

    image = _to_bgr_u8(image)
    if image is None:
        return "No Face", []

    try:
        rgb, locations = _detect_face_locations(image)
        if len(locations) == 0:
            return "No Face", []

        target_loc = _largest_face(locations)
        encodings = face_recognition.face_encodings(rgb, known_face_locations=[target_loc], num_jitters=2)

        if len(encodings) == 0:
            return "Unknown", []

        encoding = encodings[0]
        distances = face_recognition.face_distance(data["encodings"], encoding)
        if len(distances) == 0:
            return "Unknown", locations

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        # Confidence gate: best match must be under threshold and
        # clearly better than the second-best candidate.
        if len(distances) > 1:
            sorted_dist = np.sort(distances)
            distance_gap = float(sorted_dist[1] - sorted_dist[0])
        else:
            distance_gap = MIN_DISTANCE_GAP

        if best_distance <= MATCH_TOLERANCE and distance_gap >= MIN_DISTANCE_GAP:
            return data["names"][best_idx], locations
        return "Unknown", locations

    except Exception as e:
        print("Recognition Error:", e)
        return "Error", []