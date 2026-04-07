from flask import Flask, render_template, request, jsonify
import os, base64, cv2
import numpy as np

from config import DATASET_PATH
from utils.face_utils import encode_faces, recognize_face
from utils.attendance_utils import mark_attendance, get_attendance_records

app = Flask(__name__)

_LOCALHOSTS = {"127.0.0.1", "::1"}


def _is_local_request() -> bool:
    return request.remote_addr in _LOCALHOSTS


@app.route("/")
def dashboard():
    return render_template("dashboard.html", can_register=_is_local_request())


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    # Security: allow registration only from the same machine (PC),
    # so mobile users can take attendance but cannot register new faces.
    if not _is_local_request():
        return jsonify({"status": "Registration disabled on this device/network"}), 403

    if request.method == "POST":
        data = request.json

        user_id = data.get("user_id")
        name = data.get("name")
        images = data.get("images", [])

        if not user_id or not name or not images:
            return jsonify({"status": "Invalid Data"}), 400

        folder_name = f"{user_id}_{name}"
        user_path = os.path.join(DATASET_PATH, folder_name)

        os.makedirs(user_path, exist_ok=True)

        saved_count = 0

        for i, img_data in enumerate(images):
            try:
                # decode base64
                img_bytes = base64.b64decode(img_data.split(",")[1])
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 🔥 HARD FIX START
                if img is None:
                    print("Decode failed")
                    continue

                # force uint8
                img = np.array(img, dtype=np.uint8)

                # ensure 3 channel
                if len(img.shape) != 3:
                    print("Invalid shape")
                    continue

                # normalize if needed
                if img.max() <= 1:
                    img = (img * 255).astype("uint8")

                # skip too dark
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if gray.mean() < 20:
                    print("Too dark image skipped")
                    continue
                # 🔥 HARD FIX END

                save_path = os.path.join(user_path, f"{i}.jpg")
                cv2.imwrite(save_path, img)

                saved_count += 1

            except Exception as e:
                print("Image Save Error:", e)
                continue

        if saved_count == 0:
            return jsonify({"status": "No valid images captured"}), 400

        encode_faces()

        return jsonify({"status": "Registered Successfully"})

    return render_template("register.html")


# ---------------- ATTENDANCE ----------------
@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        try:
            img_data = request.json["image"]

            img_bytes = base64.b64decode(img_data.split(",")[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"status": "Invalid Image"})

            # Force expected type/shape
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            name, _ = recognize_face(img)

            if name == "No Data":
                return jsonify({"status": "No Users Registered"})

            if name not in ["Unknown", "No Face"]:
                user_id, username = name.split("_", 1)
                status = mark_attendance(user_id, username)

                return jsonify({"status": status, "name": username})

            return jsonify({"status": "Not Matched"})

        except Exception as e:
            print("Attendance Error:", e)
            return jsonify({"status": "Error Occurred"})

    return render_template("attendance.html")


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
    return jsonify(
        {
            "attendance_records": attendance_records,
            "registered_users": users,
            "total_attendance_records": len(attendance_records),
            "total_registered_users": len(users),
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
                }
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