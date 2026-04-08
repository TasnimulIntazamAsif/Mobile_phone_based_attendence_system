from flask import Flask, request, jsonify, redirect
import os, base64, cv2
import numpy as np
from datetime import datetime
import json

from config import DATASET_PATH, BASE64_REGISTRY_PATH, ENCODING_PATH, CSV_PATH
from utils.face_utils import encode_faces, recognize_face

app = Flask(__name__)


def _decode_base64_image(img_data: str):
    if not img_data or "," not in img_data:
        return None
    try:
        img_bytes = base64.b64decode(img_data.split(",", 1)[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _normalize_bgr_image(img):
    if img is None:
        return None
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


def _decode_uploaded_file_image(file_storage):
    if file_storage is None:
        return None
    try:
        data = file_storage.read()
        if not data:
            return None
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return _normalize_bgr_image(img)
    except Exception:
        return None


def _save_registration_images(user_id: str, name: str, images):
    folder_name = f"{user_id}_{name}"
    user_path = os.path.join(DATASET_PATH, folder_name)
    os.makedirs(user_path, exist_ok=True)

    saved_files = []
    for i, img_data in enumerate(images):
        img = _decode_base64_image(img_data)
        img = _normalize_bgr_image(img)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 20:
            continue

        save_path = os.path.join(user_path, f"{i}.jpg")
        if cv2.imwrite(save_path, img):
            saved_files.append(save_path)

    return user_path, saved_files


def _save_uploaded_images(user_id: str, name: str, file_storages):
    folder_name = f"{user_id}_{name}"
    user_path = os.path.join(DATASET_PATH, folder_name)
    os.makedirs(user_path, exist_ok=True)

    saved_files = []
    for i, fs in enumerate(file_storages):
        img = _decode_uploaded_file_image(fs)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 20:
            continue

        save_path = os.path.join(user_path, f"{i}.jpg")
        if cv2.imwrite(save_path, img):
            saved_files.append(save_path)
    return user_path, saved_files


def _read_base64_registry():
    if not os.path.exists(BASE64_REGISTRY_PATH):
        return []
    try:
        with open(BASE64_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _write_base64_registry(rows):
    with open(BASE64_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _append_base64_registry(employee_id: str, name: str, images):
    rows = _read_base64_registry()
    rows.append(
        {
            "employee_id": str(employee_id),
            "name": name,
            "image_count": len(images),
            "images_base64": images,
            "stored_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_base64_registry(rows)


def _delete_dataset_user(name_to_remove: str):
    if not name_to_remove:
        return False

    candidates = []
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name == name_to_remove:
            candidates.append(folder_name)
            continue
        if "_" in folder_name:
            _, n = folder_name.split("_", 1)
            if n == name_to_remove:
                candidates.append(folder_name)

    if not candidates:
        return False

    for folder_name in candidates:
        folder_path = os.path.join(DATASET_PATH, folder_name)
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except Exception:
                    pass
        try:
            os.rmdir(folder_path)
        except Exception:
            pass

    rows = _read_base64_registry()
    filtered = []
    for r in rows:
        rid = str(r.get("employee_id", ""))
        rname = str(r.get("name", ""))
        if name_to_remove == rname:
            continue
        if "_" in name_to_remove:
            maybe_id, maybe_name = name_to_remove.split("_", 1)
            if rid == str(maybe_id) and rname == str(maybe_name):
                continue
        filtered.append(r)
    if len(filtered) != len(rows):
        _write_base64_registry(filtered)

    return True


def _clear_all_data_except_base():
    any_removed = False

    if os.path.exists(CSV_PATH):
        try:
            os.remove(CSV_PATH)
            any_removed = True
        except Exception:
            pass

    if os.path.exists(ENCODING_PATH):
        try:
            os.remove(ENCODING_PATH)
            any_removed = True
        except Exception:
            pass

    if os.path.exists(BASE64_REGISTRY_PATH):
        try:
            os.remove(BASE64_REGISTRY_PATH)
            any_removed = True
        except Exception:
            pass

    # Clear dataset subfolders but keep DATASET_PATH itself
    if os.path.isdir(DATASET_PATH):
        for folder_name in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                        any_removed = True
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(folder_path)
            except Exception:
                pass

    return any_removed


@app.route("/api/face_recognition", methods=["POST"])
def api_face_recognition_file():
    img = _decode_uploaded_file_image(request.files.get("image"))
    if img is None:
        return jsonify({"status": "Invalid Image"}), 400

    matched_name, _ = recognize_face(img)
    return jsonify({"status": "Face Recognition", "matched_name": matched_name})


@app.route("/api/check_attendance", methods=["POST"])
def api_check_attendance_file():
    name = (request.form.get("name") or "").strip()
    img = _decode_uploaded_file_image(request.files.get("image"))
    if not name or img is None:
        return jsonify({"result": 0, "status": "Bad request"}), 400

    matched_name, _ = recognize_face(img)
    if matched_name in ["No Data", "Unknown", "No Face", "Error"]:
        return jsonify({"result": 0}), 200

    ok = False
    if matched_name == name:
        ok = True
    elif "_" in matched_name:
        mid, mname = matched_name.split("_", 1)
        if name == mid or name == mname:
            ok = True

    return jsonify({"result": 1 if ok else 0}), 200

@app.route("/api/train_with_multiple_image", methods=["POST"])
def api_train_with_multiple_image():
    user_id = (request.form.get("id") or "").strip()
    files = request.files.getlist("files")
    if not user_id or not files:
        return jsonify({"status": "Bad request. Missing id or not enough images."}), 400

    _, saved = _save_uploaded_images(user_id, user_id, files)
    if len(saved) == 0:
        return jsonify({"status": "Internal server error. No face detected or other errors."}), 500

    encode_faces()
    return jsonify({"status": "Faces trained successfully.", "saved_images": len(saved)}), 200



@app.route("/api/single_image_train", methods=["POST"])
def api_single_image_train():
    user_id = (request.form.get("id") or "").strip()
    files = request.files.getlist("images")
    if not user_id or not files:
        return jsonify({"status": "Bad request. Missing id or images."}), 400

    _, saved = _save_uploaded_images(user_id, user_id, files)
    if len(saved) == 0:
        return jsonify({"status": "Internal server error."}), 500

    encode_faces()
    return jsonify({"status": "Faces trained successfully.", "saved_images": len(saved)}), 200


@app.route("/api/remove_name", methods=["POST"])
def api_remove_name():
    name_to_remove = (request.form.get("name_to_remove") or "").strip()
    if not name_to_remove:
        return jsonify({"status": "Name not found or other errors."}), 400

    ok = _delete_dataset_user(name_to_remove)
    if not ok:
        return jsonify({"status": "Name not found or other errors."}), 400

    encode_faces()
    return jsonify({"status": "Name removed successfully."}), 200


@app.route("/remove_all", methods=["POST"])
def api_remove_all():
    removed = _clear_all_data_except_base()
    if not removed:
        return jsonify({"status": "No data found or other errors."}), 400
    return jsonify({"status": "Data removed successfully."}), 200


@app.route("/etc/convert_image_to_base64", methods=["POST"])
def api_convert_image_to_base64():
    img_fs = request.files.get("image")
    if img_fs is None:
        return jsonify({"status": "Invalid Image"}), 400
    try:
        raw = img_fs.read()
        if not raw:
            return jsonify({"status": "Invalid Image"}), 400
        b64 = base64.b64encode(raw).decode("utf-8")
        return jsonify({"base64": f"data:{img_fs.mimetype or 'image/jpeg'};base64,{b64}"}), 200
    except Exception:
        return jsonify({"status": "Invalid Image"}), 400


@app.route("/detection/human", methods=["POST"])
def api_detection_human():
    img = _decode_uploaded_file_image(request.files.get("file"))
    if img is None:
        return jsonify({"result": 0}), 200

    matched_name, _ = recognize_face(img)
    if matched_name in ["No Face", "Error"]:
        return jsonify({"result": 0}), 200

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    result = 1 if (brightness >= 20 and sharpness >= 20) else 0
    return jsonify({"result": int(result)}), 200


@app.route("/base64/face_recognition", methods=["POST"])
def base64_face_recognition():
    payload = request.get_json(silent=True) or {}
    img = _normalize_bgr_image(_decode_base64_image(payload.get("image", "")))
    if img is None:
        return jsonify({"status": "Invalid Image"}), 400
    matched_name, _ = recognize_face(img)
    return jsonify({"status": "Face Recognition", "matched_name": matched_name}), 200





@app.route("/base64/train", methods=["POST"])
def base64_train():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("id")
    images = payload.get("images")

    if not user_id or images is None:
        return jsonify({"status": "Bad request. Missing id or images, or id is not an integer."}), 400

    if isinstance(images, str):
        images_list = [images]
    elif isinstance(images, list):
        images_list = images
    else:
        return jsonify({"status": "Bad request. Missing id or images, or id is not an integer."}), 400

    _, saved_files = _save_registration_images(str(user_id), str(user_id), images_list)
    if len(saved_files) == 0:
        return jsonify({"status": "Internal server error"}), 500

    _append_base64_registry(str(user_id), str(user_id), images_list)
    encode_faces()
    return jsonify({"status": "Faces trained successfully.", "saved_images": len(saved_files)}), 200


# --- Documentation only (not part of the 11 functional POST APIs) ---
def _multipart_schema(required, properties):
    return {
        "type": "object",
        "required": required,
        "properties": properties,
    }


@app.route("/", methods=["GET"])
def root():
    return redirect("/api/docs")


@app.route("/api/openapi.json", methods=["GET"])
def api_openapi():
    host = request.host_url.rstrip("/")
    binary = {"type": "string", "format": "binary"}
    return jsonify(
        {
            "openapi": "3.0.3",
            "info": {
                "title": "Mobile phone based attendance / face API",
                "version": "1.0.0",
                "description": "Application API: the 11 POST operations below. GET /api/docs and this file are documentation helpers only.",
            },
            "servers": [{"url": host}],
            "paths": {
                "/api/check_attendance": {
                    "post": {
                        "summary": "Recognize face from upload and check it matches the provided name",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(
                                        ["name", "image"],
                                        {"name": {"type": "string"}, "image": binary},
                                    )
                                }
                            },
                        },
                        "responses": {"200": {"description": "1 if match else 0"}},
                    }
                },
                "/api/face_recognition": {
                    "post": {
                        "summary": "Get recognition result from an uploaded image",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(["image"], {"image": binary})
                                }
                            },
                        },
                        "responses": {"200": {"description": "Face recognition result"}},
                    }
                },
                "/api/remove_name": {
                    "post": {
                        "summary": "Remove a registered user (dataset folder + registry row)",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(
                                        ["name_to_remove"],
                                        {"name_to_remove": {"type": "string"}},
                                    )
                                }
                            },
                        },
                        "responses": {"200": {"description": "Removed"}},
                    }
                },
                "/api/single_image_train": {
                    "post": {
                        "summary": "Train from one or more uploaded images (form field images)",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(
                                        ["id", "images"],
                                        {
                                            "id": {"type": "string"},
                                            "images": {"type": "array", "items": binary},
                                        },
                                    )
                                }
                            },
                        },
                        "responses": {"200": {"description": "Trained"}},
                    }
                },
                "/api/train_with_multiple_image": {
                    "post": {
                        "summary": "Train from multiple uploaded files (form field files)",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(
                                        ["id", "files"],
                                        {
                                            "id": {"type": "string"},
                                            "files": {"type": "array", "items": binary},
                                        },
                                    )
                                }
                            },
                        },
                        "responses": {"200": {"description": "Trained"}},
                    }
                },
                "/base64/check_attendance": {
                    "post": {
                        "summary": "Check face (base64 image) against provided id",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["id", "image"],
                                        "properties": {
                                            "id": {},
                                            "image": {"type": "string", "description": "data:image/...;base64,..."},
                                        },
                                    }
                                }
                            },
                        },
                        "responses": {"200": {"description": "1 if match else 0"}},
                    }
                },
                "/base64/face_recognition": {
                    "post": {
                        "summary": "Face recognition from base64 image",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["image"],
                                        "properties": {"image": {"type": "string"}},
                                    }
                                }
                            },
                        },
                        "responses": {"200": {"description": "Face recognition result"}},
                    }
                },
                "/base64/train": {
                    "post": {
                        "summary": "Train from base64 image(s)",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["id", "images"],
                                        "properties": {
                                            "id": {},
                                            "images": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "array", "items": {"type": "string"}},
                                                ]
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "responses": {"200": {"description": "Trained"}},
                    }
                },
                "/detection/human": {
                    "post": {
                        "summary": "Simple liveness-style signal (heuristic)",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(["file"], {"file": binary})
                                }
                            },
                        },
                        "responses": {"200": {"description": "JSON with integer field result (0 or 1)"}},
                    }
                },
                "/etc/convert_image_to_base64": {
                    "post": {
                        "summary": "Convert uploaded image to a data-URL base64 string",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": _multipart_schema(["image"], {"image": binary})
                                }
                            },
                        },
                        "responses": {"200": {"description": "base64 field"}},
                    }
                },
                "/remove_all": {
                    "post": {
                        "summary": "Clear dataset, encodings, registry, attendance CSV",
                        "responses": {"200": {"description": "Cleared"}},
                    }
                },
            },
        }
    )


@app.route("/api/docs", methods=["GET"])
def api_docs():
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Face API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  </head>
  <body style="margin:0;background:#fafafa;">
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/api/openapi.json",
        dom_id: "#swagger-ui"
      });
    </script>
  </body>
</html>
"""
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)