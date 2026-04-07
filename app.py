from flask import Flask, request, jsonify, redirect
import os, base64, cv2
import numpy as np
from datetime import datetime
import json

from config import DATASET_PATH, BASE64_REGISTRY_PATH
from utils.face_utils import encode_faces, recognize_face
from utils.attendance_utils import mark_attendance, get_attendance_records

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


def _latest_record_for_user_today(user_id: str):
    today = datetime.now().strftime("%Y-%m-%d")
    records = get_attendance_records()
    for row in reversed(records):
        if str(row.get("ID", "")) == str(user_id) and str(row.get("Date", "")) == today:
            return row
    return None


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


@app.route("/", methods=["GET"])
def root():
    return redirect("/api/docs")


@app.route("/api/register-base64", methods=["POST"])
def api_register_base64():
    payload = request.get_json(silent=True) or {}
    employee_id = payload.get("employee_id")
    name = payload.get("name")
    images = payload.get("images", [])

    if not employee_id or not name or not isinstance(images, list) or len(images) == 0:
        return jsonify({"status": "Invalid Data"}), 400

    user_path, saved_files = _save_registration_images(employee_id, name, images)
    if len(saved_files) == 0:
        return jsonify({"status": "No valid images captured"}), 400

    _append_base64_registry(employee_id, name, images)
    encode_faces()
    return jsonify(
        {
            "matched": True,
            "status": "Registered and Trained",
            "employee_id": str(employee_id),
            "name": name,
            "stored_image_count": len(saved_files),
            "storage_folder": user_path,
        }
    )


@app.route("/api/v1/enroll-face", methods=["POST"])
def api_v1_enroll_face():
    payload = request.get_json(silent=True) or {}
    employee_id = payload.get("employee_id")
    employee_name = payload.get("employee_name")
    base64_images = payload.get("base64_images", [])

    if not employee_id or not employee_name or not isinstance(base64_images, list) or len(base64_images) == 0:
        return jsonify({"ok": False, "message": "employee_id, employee_name, base64_images required"}), 400

    _, saved_files = _save_registration_images(str(employee_id), str(employee_name), base64_images)
    if len(saved_files) == 0:
        return jsonify({"ok": False, "message": "No valid face image found"}), 400

    _append_base64_registry(employee_id, employee_name, base64_images)
    encode_faces()

    return jsonify(
        {
            "ok": True,
            "message": "Enrollment completed and face model refreshed",
            "employee_id": str(employee_id),
            "employee_name": str(employee_name),
            "accepted_images": len(saved_files),
            "trained": True,
        }
    )


@app.route("/api/attendance-base64", methods=["POST"])
def api_attendance_base64():
    payload = request.get_json(silent=True) or {}
    img = _normalize_bgr_image(_decode_base64_image(payload.get("image", "")))
    if img is None:
        return jsonify({"matched": False, "status": "Invalid Image"}), 400

    matched_name, _ = recognize_face(img)
    if matched_name == "No Data":
        return jsonify({"matched": False, "status": "No Users Registered"}), 404
    if matched_name in ["Unknown", "No Face", "Error"]:
        return jsonify({"matched": False, "status": "Unmatched"}), 200

    employee_id, employee_name = matched_name.split("_", 1)
    attendance_status = mark_attendance(employee_id, employee_name)
    latest = _latest_record_for_user_today(employee_id) or {}
    return jsonify(
        {
            "matched": True,
            "status": attendance_status,
            "employee_id": str(employee_id),
            "name": employee_name,
            "date": latest.get("Date", datetime.now().strftime("%Y-%m-%d")),
            "arrival": latest.get("Arrival", ""),
            "exit": latest.get("Exit", ""),
            "attendance_state": latest.get("Status", ""),
        }
    )


@app.route("/api/v1/verify-attendance", methods=["POST"])
def api_v1_verify_attendance():
    payload = request.get_json(silent=True) or {}
    image_base64 = payload.get("image_base64", "")

    img = _normalize_bgr_image(_decode_base64_image(image_base64))
    if img is None:
        return jsonify({"ok": False, "matched": False, "message": "Invalid base64 image"}), 400

    matched_name, _ = recognize_face(img)
    if matched_name == "No Data":
        return jsonify({"ok": False, "matched": False, "message": "No registered user found"}), 404
    if matched_name in ["Unknown", "No Face", "Error"]:
        return jsonify({"ok": True, "matched": False, "message": "Unmatched"}), 200

    employee_id, employee_name = matched_name.split("_", 1)
    process_status = mark_attendance(employee_id, employee_name)
    latest = _latest_record_for_user_today(employee_id) or {}

    return jsonify(
        {
            "ok": True,
            "matched": True,
            "message": process_status,
            "employee_id": str(employee_id),
            "name": employee_name,
            "date": latest.get("Date", datetime.now().strftime("%Y-%m-%d")),
            "arrival": latest.get("Arrival", ""),
            "exit": latest.get("Exit", ""),
            "status": latest.get("Status", ""),
        }
    )


def _get_registered_users():
    users = []
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if "_" in folder_name:
            user_id, name = folder_name.split("_", 1)
        else:
            user_id, name = folder_name, folder_name
        users.append({"id": user_id, "name": name})
    return users


@app.route("/api/stored-data", methods=["GET"])
def api_stored_data():
    attendance_records = get_attendance_records()
    users = _get_registered_users()
    base64_registry = _read_base64_registry()
    return jsonify(
        {
            "attendance_records": attendance_records,
            "registered_users": users,
            "base64_registry": base64_registry,
            "total_attendance_records": len(attendance_records),
            "total_registered_users": len(users),
            "total_base64_registry_entries": len(base64_registry),
        }
    )


@app.route("/api/openapi.json", methods=["GET"])
def api_openapi():
    host = request.host_url.rstrip("/")
    return jsonify(
        {
            "openapi": "3.0.3",
            "info": {
                "title": "Mobile Attendance System API",
                "version": "1.0.0",
                "description": "API to inspect stored attendance and registered user data.",
            },
            "servers": [{"url": host}],
            "paths": {
                "/api/stored-data": {
                    "get": {
                        "summary": "Get stored system data",
                        "description": "Returns attendance records from CSV and registered users from dataset folders.",
                        "responses": {
                            "200": {
                                "description": "Stored data fetched successfully",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "attendance_records": {
                                                    "type": "array",
                                                    "items": {"$ref": "#/components/schemas/AttendanceRecord"},
                                                },
                                                "registered_users": {
                                                    "type": "array",
                                                    "items": {"$ref": "#/components/schemas/RegisteredUser"},
                                                },
                                                "total_attendance_records": {"type": "integer"},
                                                "total_registered_users": {"type": "integer"},
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
                "/api/register-base64": {
                    "post": {
                        "summary": "Register user using base64 images",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RegisterBase64Request"}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "User registered and model encoded",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/RegisterBase64Response"}
                                    }
                                },
                            }
                        },
                    }
                },
                "/api/attendance-base64": {
                    "post": {
                        "summary": "Match face and mark attendance",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AttendanceBase64Request"}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Face match attempt result",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/AttendanceBase64Response"}
                                    }
                                },
                            }
                        },
                    }
                },
                "/api/v1/enroll-face": {
                    "post": {
                        "summary": "Enroll face from base64 image list",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/EnrollFaceRequest"}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Enrollment and training success",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/EnrollFaceResponse"}
                                    }
                                },
                            }
                        },
                    }
                },
                "/api/v1/verify-attendance": {
                    "post": {
                        "summary": "Verify face and store attendance",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VerifyAttendanceRequest"}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Matched/unmatched result with attendance fields",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/VerifyAttendanceResponse"}
                                    }
                                },
                            }
                        },
                    }
                },
            },
            "components": {
                "schemas": {
                    "AttendanceRecord": {
                        "type": "object",
                        "properties": {
                            "ID": {"type": "string", "example": "570508"},
                            "Name": {"type": "string", "example": "Asif"},
                            "Date": {"type": "string", "example": "2026-04-07"},
                            "Arrival": {"type": "string", "example": "13:22:08"},
                            "Exit": {"type": "string", "example": "17:35:41"},
                            "Status": {"type": "string", "example": "Exited"},
                        },
                    },
                    "RegisteredUser": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "example": "570508"},
                            "name": {"type": "string", "example": "Asif"},
                        },
                    },
                    "RegisterBase64Request": {
                        "type": "object",
                        "required": ["employee_id", "name", "images"],
                        "properties": {
                            "employee_id": {"type": "string", "example": "570508"},
                            "name": {"type": "string", "example": "Asif"},
                            "images": {
                                "type": "array",
                                "description": "List of base64 image strings (data:image/jpeg;base64,...)",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "RegisterBase64Response": {
                        "type": "object",
                        "properties": {
                            "matched": {"type": "boolean", "example": True},
                            "status": {"type": "string", "example": "Registered and Trained"},
                            "employee_id": {"type": "string", "example": "570508"},
                            "name": {"type": "string", "example": "Asif"},
                            "stored_image_count": {"type": "integer", "example": 5},
                            "storage_folder": {"type": "string"},
                        },
                    },
                    "AttendanceBase64Request": {
                        "type": "object",
                        "required": ["image"],
                        "properties": {
                            "image": {
                                "type": "string",
                                "description": "Single base64 image string (data:image/jpeg;base64,...)",
                            }
                        },
                    },
                    "AttendanceBase64Response": {
                        "type": "object",
                        "properties": {
                            "matched": {"type": "boolean"},
                            "status": {"type": "string"},
                            "employee_id": {"type": "string"},
                            "name": {"type": "string"},
                            "date": {"type": "string"},
                            "arrival": {"type": "string"},
                            "exit": {"type": "string"},
                            "attendance_state": {"type": "string"},
                        },
                    },
                    "EnrollFaceRequest": {
                        "type": "object",
                        "required": ["employee_id", "employee_name", "base64_images"],
                        "properties": {
                            "employee_id": {"type": "string", "example": "570508"},
                            "employee_name": {"type": "string", "example": "Asif"},
                            "base64_images": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Multiple base64 images (data:image/jpeg;base64,...)",
                            },
                        },
                    },
                    "EnrollFaceResponse": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean", "example": True},
                            "message": {"type": "string"},
                            "employee_id": {"type": "string"},
                            "employee_name": {"type": "string"},
                            "accepted_images": {"type": "integer"},
                            "trained": {"type": "boolean", "example": True},
                        },
                    },
                    "VerifyAttendanceRequest": {
                        "type": "object",
                        "required": ["image_base64"],
                        "properties": {
                            "image_base64": {"type": "string"},
                        },
                    },
                    "VerifyAttendanceResponse": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean"},
                            "matched": {"type": "boolean"},
                            "message": {"type": "string"},
                            "employee_id": {"type": "string"},
                            "name": {"type": "string"},
                            "date": {"type": "string"},
                            "arrival": {"type": "string"},
                            "exit": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                }
            },
        }
    )


@app.route("/api/docs", methods=["GET"])
def api_docs():
    # Lightweight Swagger UI without extra Python dependency
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Attendance API Docs</title>
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